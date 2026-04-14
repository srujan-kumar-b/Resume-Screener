import os
import joblib

from model_training import main as train_model


DATASET_PATH = "cleaned_resume_screener_dataset.csv"
MODEL_PATH = "resume_role_model.pkl"


def print_saved_accuracy(model_path: str) -> None:
    if not os.path.exists(model_path):
        print("Model file not found after training.")
        return

    bundle = joblib.load(model_path)
    accuracy = bundle.get("accuracy") if isinstance(bundle, dict) else None

    if isinstance(accuracy, (int, float)):
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
    else:
        print("Model Accuracy: unavailable in model file")


def main() -> None:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Please generate or place it before training."
        )

    print(f"Using dataset: {DATASET_PATH}")
    train_model()
    print_saved_accuracy(MODEL_PATH)


if __name__ == "__main__":
    main()
