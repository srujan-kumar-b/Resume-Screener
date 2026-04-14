from io import BytesIO
import os
import re

from flask import Flask, render_template, request
import joblib

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

app = Flask(__name__)

MODEL_PATH = "resume_role_model.pkl"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}

SKILLS_LIST = [
    "python", "java", "sql", "machine learning", "deep learning", "nlp", "data analysis",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "html", "css", "javascript",
    "react", "node", "flask", "django", "aws", "azure", "docker", "kubernetes", "git",
    "power bi", "tableau", "spark", "hadoop", "excel"
]
JD_SKILLS = ["python", "sql", "machine learning", "aws", "nlp"]


def load_model_bundle(path):
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle

    # Backward compatibility: if only model object was saved.
    return {
        "model": bundle,
        "feature_columns": [],
        "text_columns": [],
        "numeric_columns": [],
        "target_column": "predicted_role",
    }


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(file_storage):
    filename = file_storage.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    data = file_storage.read()

    if ext == "txt":
        return data.decode("utf-8", errors="ignore")

    if ext == "pdf":
        if PdfReader is None:
            raise RuntimeError("PDF support is unavailable. Please install pypdf.")
        reader = PdfReader(BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if ext == "docx":
        if Document is None:
            raise RuntimeError("DOCX support is unavailable. Please install python-docx.")
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    if ext in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}:
        if Image is None or pytesseract is None:
            raise RuntimeError(
                "Image OCR support is unavailable. Install pillow and pytesseract, "
                "and ensure Tesseract OCR is installed on your system."
            )
        image = Image.open(BytesIO(data))
        return pytesseract.image_to_string(image)

    raise ValueError("Unsupported file type. Upload image, PDF, DOCX, or TXT.")


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_skills(text):
    text_l = text.lower()
    found = []
    for skill in SKILLS_LIST:
        if re.search(rf"\b{re.escape(skill)}\b", text_l):
            found.append(skill)
    return sorted(set(found))


def experience_level(text):
    years = re.findall(r"(\d{1,2})\+?\s*(?:years|year|yrs|yr)", text.lower())
    if not years:
        return "Unknown"
    max_year = max(int(y) for y in years)
    if max_year < 1:
        return "Fresher"
    if max_year < 3:
        return "Intermediate"
    return "Experienced"


def compute_scores(text, skills_found):
    t = text.lower()
    section_keywords = {
        "education": ["education", "b.tech", "bachelor", "master", "degree", "university"],
        "experience": ["experience", "worked", "intern", "employment", "project"],
        "certifications": ["certification", "certified", "course"],
        "achievements": ["achievement", "award", "recognition"],
        "languages": ["language", "english", "hindi", "telugu", "tamil"],
        "skills": ["skills", "tools", "technologies", "stack"],
    }

    hit_count = 0
    for keywords in section_keywords.values():
        if any(k in t for k in keywords):
            hit_count += 1

    word_count = len(text.split())
    completeness_score = (hit_count / len(section_keywords)) * 70 + min(word_count / 350, 1) * 30

    overlap = len(set(skills_found) & set(JD_SKILLS))
    match_score = (overlap / len(JD_SKILLS)) * 100 if JD_SKILLS else 0.0
    final_score = (match_score * 0.7) + (completeness_score * 0.3)

    return round(completeness_score, 2), round(match_score, 2), round(final_score, 2)


def build_features_for_model(bundle, resume_text, skills_found, missing_skills, rec_text, gaps_text):
    feature_columns = bundle.get("feature_columns", [])
    text_columns = bundle.get("text_columns", [])
    numeric_columns = bundle.get("numeric_columns", [])

    completeness_score, match_score, final_score = compute_scores(resume_text, skills_found)

    values = {
        "Education": resume_text,
        "Experience": resume_text,
        "Certifications": resume_text,
        "Achievements": resume_text,
        "Languages": resume_text,
        "Interests": resume_text,
        "skills_extracted": ", ".join(skills_found),
        "missing_skills": ", ".join(missing_skills),
        "resume_gaps": gaps_text,
        "recommendations": rec_text,
        "__combined_text__": resume_text,
        "completeness_score": completeness_score,
        "match_score": match_score,
        "final_score": final_score,
    }

    row = {}
    for col in text_columns:
        row[col] = values.get(col, resume_text)
    for col in numeric_columns:
        row[col] = float(values.get(col, 0.0))

    # Fallback for older model bundles without explicit schema metadata.
    if not feature_columns:
        feature_columns = list(row.keys())

    if "__combined_text__" not in feature_columns:
        feature_columns = list(feature_columns) + ["__combined_text__"]

    data = {}
    for col in feature_columns:
        default_value = "" if (col in text_columns or col == "__combined_text__") else 0.0
        data[col] = [row.get(col, default_value)]

    return data, completeness_score, match_score, final_score


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}. Run model_training.py first.")

model_bundle = load_model_bundle(MODEL_PATH)
model = model_bundle["model"]
model_accuracy = model_bundle.get("accuracy")

if isinstance(model_accuracy, (int, float)):
    print(f"Loaded model accuracy: {model_accuracy * 100:.2f}%")
else:
    print("Loaded model accuracy: unavailable")


def get_prediction_confidence(model, features, predicted_label):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(getattr(model, "classes_", []))
        if classes and predicted_label in classes:
            return float(probabilities[classes.index(predicted_label)]) * 100.0
        return float(max(probabilities)) * 100.0

    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        if getattr(scores, "ndim", 1) > 1:
            scores = scores[0]
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            else:
                scores = list(scores)
            classes = list(getattr(model, "classes_", []))
            if classes and predicted_label in classes:
                return float(scores[classes.index(predicted_label)])
            return float(max(scores))

    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    name = request.form.get('name', '').strip() or "Candidate"
    resume_file = request.files.get('resume')

    if not resume_file or not resume_file.filename:
        return render_template('result.html', error="Please upload a resume file.")

    if not allowed_file(resume_file.filename):
        return render_template('result.html', error="Unsupported file type. Upload image, PDF, DOCX, or TXT.")

    try:
        resume_text = clean_text(extract_text(resume_file))
    except Exception as exc:
        return render_template('result.html', error=f"Could not read uploaded file: {exc}")

    if not resume_text:
        return render_template('result.html', error="Uploaded file is empty or has no extractable text.")

    skills_found = extract_skills(resume_text)
    missing_skills = sorted(set(JD_SKILLS) - set(skills_found))

    gap_items = []
    if "certification" not in resume_text.lower() and "certified" not in resume_text.lower():
        gap_items.append("No certifications found")
    if "achievement" not in resume_text.lower() and "award" not in resume_text.lower():
        gap_items.append("No achievements found")
    if not re.search(r"\d{1,2}\+?\s*(years|year|yrs|yr)", resume_text.lower()):
        gap_items.append("No clear experience duration found")

    recommendations = []
    if missing_skills:
        recommendations.append("Add projects using: " + ", ".join(missing_skills[:3]))
    if "No certifications found" in gap_items:
        recommendations.append("Add at least 1 relevant certification")
    if "No achievements found" in gap_items:
        recommendations.append("Highlight measurable achievements")

    feature_data, completeness_score, match_score, final_score = build_features_for_model(
        model_bundle,
        resume_text,
        skills_found,
        missing_skills,
        "; ".join(recommendations) if recommendations else "Profile looks strong. Keep tailoring for each role.",
        "; ".join(gap_items) if gap_items else "No major gaps detected",
    )

    import pandas as pd
    pred_df = pd.DataFrame(feature_data)
    role = model.predict(pred_df)[0]
    prediction_confidence = get_prediction_confidence(model, pred_df, role)

    return render_template('result.html',
                           name=name,
                           role=role,
                           prediction_confidence=round(prediction_confidence, 2) if isinstance(prediction_confidence, (int, float)) else None,
                           model_accuracy=round(model_accuracy * 100, 2) if isinstance(model_accuracy, (int, float)) else None,
                           match_score=match_score,
                           final_score=final_score,
                           completeness_score=completeness_score,
                           skills=", ".join(skills_found) if skills_found else "No major skills detected",
                           missing=", ".join(missing_skills) if missing_skills else "None",
                           exp=experience_level(resume_text),
                           gaps="; ".join(gap_items) if gap_items else "No major gaps detected",
                           rec="; ".join(recommendations) if recommendations else "Profile looks strong. Keep tailoring for each role.")

if __name__ == '__main__':
    app.run(debug=True)