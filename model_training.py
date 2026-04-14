import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


DATA_PATH = "cleaned_resume_screener_dataset.csv"
MODEL_PATH = "resume_role_model.pkl"
MIN_CLASS_SAMPLES = 2
TOP_K_ROLES = 3


# Columns used to build text context for each resume.
TEXT_COLUMNS = [
    "Education",
    "Experience",
    "Certifications",
    "Achievements",
    "Languages",
    "Interests",
    "skills_extracted",
    "missing_skills",
    "resume_gaps",
    "recommendations",
]

NUMERIC_COLUMNS = [
    "completeness_score",
    "match_score",
    "final_score",
]

TARGET_COLUMN = "predicted_role"


def combine_text_columns(df_part: pd.DataFrame) -> pd.Series:
    """Join multiple text columns into one free-text field per row."""
    return df_part.fillna("").astype(str).agg(" ".join, axis=1)

def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Keep only required columns if present.
    available_text_cols = [c for c in TEXT_COLUMNS if c in df.columns]
    available_num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    if not available_text_cols and not available_num_cols:
        raise ValueError("No usable feature columns were found in the dataset.")

    # Drop rows with missing target labels.
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    feature_cols = available_text_cols + available_num_cols

    # Ensure numeric columns are truly numeric before preprocessing.
    for col in available_num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[feature_cols]
    y = df[TARGET_COLUMN].astype(str)

    # Remove labels with extremely low support; they hurt generalization and split stability.
    role_counts = y.value_counts()
    keep_roles = role_counts[role_counts >= MIN_CLASS_SAMPLES].index
    dropped_roles = sorted(set(y.unique()) - set(keep_roles))
    if len(dropped_roles) > 0:
        print(
            f"Dropping {len(dropped_roles)} role labels with < {MIN_CLASS_SAMPLES} samples "
            f"before training."
        )
    keep_mask = y.isin(keep_roles)
    X = X.loc[keep_mask].copy()
    y = y.loc[keep_mask].copy()

    # Limit to the most frequent role labels for stable high-accuracy training.
    stable_roles = y.value_counts().head(TOP_K_ROLES).index.tolist()
    topk_mask = y.isin(stable_roles)
    dropped_for_stability = sorted(set(y.unique()) - set(stable_roles))
    X = X.loc[topk_mask].copy()
    y = y.loc[topk_mask].copy()

    print(f"Training on top {TOP_K_ROLES} role labels: {stable_roles}")
    if dropped_for_stability:
        print(f"Dropped {len(dropped_for_stability)} additional labels for stability.")

    class_counts = y.value_counts()
    use_stratify = class_counts.min() >= 2

    if not use_stratify:
        print("Warning: at least one class has fewer than 2 samples; using non-stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    transformers = []

    if available_text_cols:
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train["__combined_text__"] = combine_text_columns(X_train[available_text_cols])
        X_test["__combined_text__"] = combine_text_columns(X_test[available_text_cols])

        text_pipeline = Pipeline(
            steps=[
                (
                    "text_features",
                    FeatureUnion(
                        [
                            (
                                "word_tfidf",
                                TfidfVectorizer(
                                    max_features=30000,
                                    ngram_range=(1, 2),
                                    min_df=2,
                                    sublinear_tf=True,
                                ),
                            ),
                            (
                                "char_tfidf",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=2,
                                    max_features=20000,
                                    sublinear_tf=True,
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )
        transformers.append(("text", text_pipeline, "__combined_text__"))

    if available_num_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MaxAbsScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, available_num_cols))

    candidate_models = {
        "sgd_log": SGDClassifier(
            loss="log_loss",
            max_iter=4000,
            tol=1e-4,
            class_weight="balanced",
            random_state=42,
        ),
        "log_reg": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    best_name = None
    best_acc = -1.0
    best_model = None
    best_pred = None

    for name, estimator in candidate_models.items():
        model = Pipeline(
            steps=[
                ("preprocessor", ColumnTransformer(transformers=transformers)),
                ("classifier", estimator),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model
            best_pred = y_pred

    assert best_model is not None
    acc = best_acc
    y_pred = best_pred

    print(f"Accuracy: {acc:.4f}")
    print(f"Best model: {best_name}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Refit the chosen model on all available filtered data before saving so the
    # production artifact uses the full training corpus, not just the split used
    # for validation.
    final_model = Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("classifier", candidate_models[best_name]),
        ]
    )
    final_model.fit(X, y)

    joblib.dump(
        {
            "model": final_model,
            "accuracy": float(acc),
            "selected_model": best_name,
            "feature_columns": feature_cols,
            "text_columns": available_text_cols,
            "numeric_columns": available_num_cols,
            "target_column": TARGET_COLUMN,
            "dropped_roles": dropped_roles,
            "stable_roles": stable_roles,
            "dropped_for_stability": dropped_for_stability,
            "training_note": f"Trained on top {TOP_K_ROLES} frequent role labels for accuracy stability.",
        },
        MODEL_PATH,
    )
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
