from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from string import punctuation

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import h5py
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from chatbot import get_response_with_engine as chatbot_get_response


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
MODEL_DIR = BASE_DIR / "model"
DEFAULT_TEST_DATA = DATASET_DIR / "testData.csv"
TRAINING_DATA = (
    DATASET_DIR / "india_training_dataset.csv"
    if (DATASET_DIR / "india_training_dataset.csv").exists()
    else DATASET_DIR / "fake_job_postings.csv"
)
SAVED_BERT_FEATURES = MODEL_DIR / "bert_X.npy"
CNN_WEIGHTS = MODEL_DIR / "cnn2d_weights.hdf5"
SAVED_LABELS = MODEL_DIR / "Y.npy"

LABELS = ["Real Job", "Fraudulent Job"]
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


app = Flask(__name__)
app.secret_key = "welcome"
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

PUBLIC_ENDPOINTS = {"root", "admin_login", "admin_login_action", "static", "chat", "chat_status"}


class NumpyCnn2D:
    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.weights = weights

    @classmethod
    def load(cls, path: Path) -> "NumpyCnn2D":
        with h5py.File(path, "r") as handle:
            weights = {
                "conv1_kernel": handle["model_weights/conv2d_1/conv2d_1/kernel:0"][:],
                "conv1_bias": handle["model_weights/conv2d_1/conv2d_1/bias:0"][:],
                "conv2_kernel": handle["model_weights/conv2d_2/conv2d_2/kernel:0"][:],
                "conv2_bias": handle["model_weights/conv2d_2/conv2d_2/bias:0"][:],
                "dense1_kernel": handle["model_weights/dense_13/dense_13/kernel:0"][:],
                "dense1_bias": handle["model_weights/dense_13/dense_13/bias:0"][:],
                "dense2_kernel": handle["model_weights/dense_14/dense_14/kernel:0"][:],
                "dense2_bias": handle["model_weights/dense_14/dense_14/bias:0"][:],
            }
        return cls(weights)

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = features.reshape(features.shape[0], 32, 24, 1).astype(np.float32)
        x = relu(conv2d_valid(x, self.weights["conv1_kernel"], self.weights["conv1_bias"]))
        x = max_pool2d(x)
        x = relu(conv2d_valid(x, self.weights["conv2_kernel"], self.weights["conv2_bias"]))
        x = max_pool2d(x)
        x = x.reshape(x.shape[0], -1)
        x = relu(x @ self.weights["dense1_kernel"] + self.weights["dense1_bias"])
        logits = x @ self.weights["dense2_kernel"] + self.weights["dense2_bias"]
        return softmax(logits)


def conv2d_valid(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    kh, kw, _, _ = kernel.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))
    return np.tensordot(windows, kernel, axes=([3, 4, 5], [2, 0, 1])) + bias


def max_pool2d(x: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(x, (2, 2), axis=(1, 2))
    windows = windows[:, ::2, ::2, :, :, :]
    return windows.max(axis=(4, 5))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@lru_cache(maxsize=1)
def load_cnn_model() -> NumpyCnn2D:
    return NumpyCnn2D.load(CNN_WEIGHTS)


@lru_cache(maxsize=1)
def load_training_index() -> pd.DataFrame:
    training = pd.read_csv(TRAINING_DATA)
    training = training.reset_index().rename(columns={"index": "_feature_index"})
    training = training[["_feature_index", "job_id", "description"]].copy()
    training["job_id"] = training["job_id"].astype(str)
    return training


@lru_cache(maxsize=1)
def load_saved_bert_features() -> np.ndarray:
    return np.load(SAVED_BERT_FEATURES)


@lru_cache(maxsize=1)
def project_metrics() -> dict[str, str]:
    labels = np.load(SAVED_LABELS)
    real_count = int(np.sum(labels == 0))
    fraud_count = int(np.sum(labels == 1))
    total = int(labels.shape[0])
    fraud_rate = (fraud_count / total) * 100 if total else 0
    return {
        "total_records": f"{total:,}",
        "real_jobs": f"{real_count:,}",
        "fraud_jobs": f"{fraud_count:,}",
        "fraud_rate": f"{fraud_rate:.2f}%",
    }


def clean_text(doc: str) -> str:
    tokens = str(doc).strip().lower().split()
    table = str.maketrans("", "", punctuation)
    tokens = [word.translate(table) for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in STOP_WORDS]
    tokens = [word for word in tokens if len(word) > 1]
    return " ".join(tokens)


def embeddings_from_saved_features(data: pd.DataFrame) -> np.ndarray | None:
    training = load_training_index()
    saved_features = load_saved_bert_features()
    saved_size = saved_features.shape[0]

    def valid_saved_indices(frame: pd.DataFrame) -> np.ndarray | None:
        if frame["_feature_index"].isna().any():
            return None
        indices = frame["_feature_index"].astype(int).to_numpy()
        if indices.size == 0 or indices.max() >= saved_size or indices.min() < 0:
            return None
        return indices

    if "job_id" in data.columns:
        job_keys = data[["job_id"]].copy()
        job_keys["job_id"] = job_keys["job_id"].astype(str)
        merged = job_keys.merge(training, on="job_id", how="left")
        indices = valid_saved_indices(merged)
        if indices is not None:
            return saved_features[indices]

    merged = data[["description"]].merge(training, on="description", how="left")
    indices = valid_saved_indices(merged)
    if indices is not None:
        return saved_features[indices]

    return None


@lru_cache(maxsize=1)
def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")


def build_embeddings(data: pd.DataFrame) -> np.ndarray:
    saved_features = embeddings_from_saved_features(data)
    if saved_features is not None:
        return saved_features

    cleaned = [clean_text(text) for text in data["description"].fillna("")]
    return np.asarray(load_sentence_transformer().encode(cleaned), dtype=np.float32)


def load_prediction_data() -> pd.DataFrame:
    upload = request.files.get("t1")
    manual_description = request.form.get("manual_description", "").strip()

    if request.form.get("manual_check") == "1":
        if not manual_description:
            raise ValueError("Please enter a job description for manual checking.")
        data = pd.DataFrame(
            [
                {
                    "job_id": "manual-entry",
                    "title": request.form.get("manual_title", "").strip() or "Manual Job Entry",
                    "location": request.form.get("manual_location", "").strip() or "Not specified",
                    "description": manual_description,
                }
            ]
        )
    elif upload and upload.filename:
        data = pd.read_csv(upload)
    else:
        data = pd.read_csv(DEFAULT_TEST_DATA)

    if "description" not in data.columns:
        raise ValueError("The CSV must include a 'description' column.")

    return data


def predict_jobs(data: pd.DataFrame) -> np.ndarray:
    features = build_embeddings(data)
    model = load_cnn_model()
    return model.predict(features)


def build_prediction_results(
    data: pd.DataFrame, probabilities: np.ndarray
) -> list[dict[str, object]]:
    results = []
    for index, (_, row) in enumerate(data.fillna("").iterrows(), start=1):
        scores = probabilities[index - 1]
        prediction = int(np.argmax(scores))
        confidence = float(scores[prediction]) * 100
        results.append(
            {
                "serial": index,
                "title": row.get("title") or f"Job Posting {index}",
                "location": row.get("location") or "Not specified",
                "description": row.get("description") or "No description available",
                "prediction": LABELS[prediction],
                "confidence": f"{confidence:.2f}%",
                "confidence_value": round(confidence, 2),
                "is_fraud": prediction == 1,
            }
        )
    return results


def prediction_summary(results: list[dict[str, object]]) -> dict[str, object]:
    fraud_count = sum(1 for result in results if result["is_fraud"])
    real_count = len(results) - fraud_count
    avg_confidence = (
        round(sum(float(result["confidence_value"]) for result in results) / len(results), 2)
        if results
        else 0.0
    )
    high_risk_count = sum(
        1 for result in results if result["is_fraud"] and float(result["confidence_value"]) >= 85
    )
    return {
        "total": len(results),
        "real": real_count,
        "fraud": fraud_count,
        "avg_confidence": avg_confidence,
        "fraud_ratio": round((fraud_count / len(results)) * 100, 2) if results else 0.0,
        "high_risk": high_risk_count,
    }


def build_risk_insights(
    results: list[dict[str, object]], summary: dict[str, object]
) -> list[dict[str, str]]:
    fraud_locations = [
        str(result["location"])
        for result in results
        if result["is_fraud"] and str(result["location"]) != "Not specified"
    ]
    top_fraud_location = (
        Counter(fraud_locations).most_common(1)[0][0] if fraud_locations else "Mixed / Unspecified"
    )

    if summary["fraud"] == 0:
        location_message = "No fraudulent rows detected in this batch."
    else:
        location_message = f"Most flagged jobs are clustered around {top_fraud_location}."

    return [
        {
            "title": "Fraud Pressure",
            "value": f"{summary['fraud_ratio']:.2f}%",
            "detail": "Share of uploaded jobs flagged as suspicious.",
            "tone": "alert" if summary["fraud"] else "calm",
        },
        {
            "title": "High-Risk Alerts",
            "value": str(summary["high_risk"]),
            "detail": "Flagged records above 85% confidence.",
            "tone": "alert" if summary["high_risk"] else "calm",
        },
        {
            "title": "Location Watch",
            "value": top_fraud_location,
            "detail": location_message,
            "tone": "info",
        },
    ]


def is_authenticated() -> bool:
    return session.get("is_admin_logged_in") is True


@app.before_request
def protect_private_routes():
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if not is_authenticated():
        if request.endpoint == "chat":
            return jsonify({"reply": "Please login first to use the assistant."}), 401
        return redirect(url_for("admin_login", msg="Please login to continue."))
    return None


@app.route("/")
def root():
    if is_authenticated():
        return redirect(url_for("index"))
    return redirect(url_for("admin_login"))


@app.route("/index", methods=["GET", "POST"])
def index():
    return render_template("index.html", metrics=project_metrics())


@app.route("/Predict", methods=["GET", "POST"])
def predict_view():
    return render_template("Predict.html", msg="")


@app.route("/Profile", methods=["GET"])
def profile():
    return render_template("Profile.html", admin_user=session.get("admin_user", "admin"))


@app.route("/Chatbot", methods=["GET"])
def chatbot_page():
    return render_template("Chatbot.html")


@app.route("/AdminLogin", methods=["GET", "POST"])
def admin_login():
    if is_authenticated():
        return redirect(url_for("index"))
    return render_template("AdminLogin.html", msg=request.args.get("msg", ""))


@app.route("/AdminLoginAction", methods=["GET", "POST"])
def admin_login_action():
    if request.method == "POST" and "t1" in request.form and "t2" in request.form:
        user = request.form["t1"]
        password = request.form["t2"]
        if user == "admin" and password == "admin":
            session["is_admin_logged_in"] = True
            session["admin_user"] = user
            return redirect(url_for("index"))
    return render_template("AdminLogin.html", msg="Invalid login details")


@app.route("/Logout")
def logout():
    session.clear()
    return redirect(url_for("admin_login", msg="Logged out successfully."))


@app.route("/PredictAction", methods=["GET", "POST"])
def predict_action():
    if request.method != "POST":
        return render_template("Predict.html", msg="")

    try:
        data = load_prediction_data()
        probabilities = predict_jobs(data)
        results = build_prediction_results(data, probabilities)
        summary = prediction_summary(results)
        risk_insights = build_risk_insights(results, summary)
    except Exception as exc:
        return render_template("AdminScreen.html", error=f"Prediction failed: {exc}")

    return render_template(
        "AdminScreen.html",
        results=results,
        summary=summary,
        risk_insights=risk_insights,
    )


@app.route("/chat", methods=["POST"])
def chat():
    """AI chatbot endpoint.

    Accepts JSON:
        { "message": str, "history": [{"role": "user"|"model", "content": str}, ...] }

    Returns JSON:
        { "reply": str, "engine": "gemini"|"fallback" }
    """
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()
    history = data.get("history", [])

    # Validate history is a list of dicts
    if not isinstance(history, list):
        history = []

    reply, engine = chatbot_get_response(user_message, history)

    payload = {"reply": reply, "engine": engine}
    if engine != "gemini":
        try:
            from chatbot import get_gemini_status
            payload.update(get_gemini_status())
        except Exception:
            pass
    return jsonify(payload)


@app.route("/chat/status", methods=["GET"])
def chat_status():
    """Return Gemini chatbot status without exposing secrets."""
    from chatbot import get_gemini_status
    return jsonify(get_gemini_status())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)




