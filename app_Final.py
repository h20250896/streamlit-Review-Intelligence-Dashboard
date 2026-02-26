import html
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix, hstack


MODEL_FILE = Path("helpfulness_model.pkl")
VECTORIZER_FILE = Path("helpfulness_vectorizer.pkl")

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
LONG_REVIEW_MIN_WORDS = 20
EXPECTED_FEATURE_COUNT = 5001
EXPECTED_TFIDF_COUNT = 5000

DATASET_STATS = {
    "raw_rows": 568454,
    "valid_rows": 298402,
    "training_sample": 100000,
    "helpful_pct": 75.45,
    "not_helpful_pct": 24.55,
    "feature_count": 5001,
}


st.set_page_config(
    page_title="Review Intelligence Dashboard",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
        }
        .app-header {
            padding: 1rem 1.25rem;
            border-radius: 12px;
            background: #ffffff;
            border: 1px solid #dde5ef;
        }
        .subtle-text {
            color: #4a5568;
            margin-top: 0.25rem;
        }
        .review-card {
            padding: 15px;
            border-radius: 10px;
            background: #f7f7f7;
            border: 1px solid #d6deea;
            margin-bottom: 0.75rem;
        }
        .history-card {
            padding: 12px;
            border-radius: 10px;
            background: #ffffff;
            border: 1px solid #d6deea;
            margin-bottom: 0.5rem;
        }
        .badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .badge-helpful {
            background: #def7e3;
            color: #0f5132;
        }
        .badge-not-helpful {
            background: #fde2e2;
            color: #842029;
        }
        .meta {
            color: #334155;
            font-size: 0.9rem;
            margin-top: 0.45rem;
        }
        .insight-box {
            padding: 0.95rem 1rem;
            border: 1px solid #d6deea;
            border-radius: 10px;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model() -> tuple[object | None, object | None, str | None]:
    try:
        if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
            return None, None, (
                "Model artifacts not found. Expected files in app directory: "
                f"{MODEL_FILE.name}, {VECTORIZER_FILE.name}."
            )

        loaded_model = joblib.load(MODEL_FILE)
        loaded_vectorizer = joblib.load(VECTORIZER_FILE)

        if not hasattr(loaded_vectorizer, "vocabulary_"):
            return None, None, "Vectorizer is missing vocabulary_ and is not usable."

        vocab_size = len(loaded_vectorizer.vocabulary_)
        model_features = getattr(loaded_model, "n_features_in_", None)
        if model_features is None:
            return None, None, "Model is missing n_features_in_ compatibility metadata."

        if vocab_size + 1 != model_features:
            return None, None, (
                "Feature mismatch detected. Expected vectorizer_vocab + 1 == "
                f"model_features, got {vocab_size} + 1 != {model_features}."
            )

        return loaded_model, loaded_vectorizer, None
    except Exception as exc:
        return None, None, f"Error loading model artifacts: {exc}"


def preprocess_text(text: str) -> str:
    cleaned = str(text).lower()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    cleaned = re.sub(r"\d+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def predict_helpfulness(review_text: str, threshold: float) -> dict | None:
    cleaned_text = preprocess_text(review_text)
    if not cleaned_text:
        return None

    tfidf_feature = vectorizer.transform([cleaned_text])
    review_length = len(cleaned_text.split())
    length_feature = csr_matrix(np.array([[review_length]], dtype=float))
    features = hstack([tfidf_feature, length_feature], format="csr")

    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Unexpected feature width {features.shape[1]}; expected "
            f"{EXPECTED_FEATURE_COUNT}."
        )

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction])
    label = "Helpful" if prediction == 1 else "Not Helpful"

    return {
        "review_text": review_text,
        "cleaned_text": cleaned_text,
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "word_count": review_length,
        "is_low_confidence": confidence < threshold,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def init_session_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None


def render_prediction_card(record: dict, threshold: float) -> None:
    is_low_confidence = record["confidence"] < threshold
    badge_class = "badge-helpful" if record["prediction"] == 1 else "badge-not-helpful"
    badge_icon = "&#9989;" if record["prediction"] == 1 else "&#10060;"
    safe_review = html.escape(record["review_text"])

    st.markdown(
        f"""
        <div class="review-card">
            <div><strong>&#128100; Customer Review</strong></div>
            <div style="margin-top:0.4rem;">{safe_review}</div>
            <div style="margin-top:0.65rem;">
                <span class="badge {badge_class}">{badge_icon} {record["label"]}</span>
            </div>
            <div class="meta">
                Confidence: {record["confidence"] * 100:.2f}% |
                Word Count: {record["word_count"]} |
                Low Confidence: {"Yes" if is_low_confidence else "No"} |
                Timestamp: {record["timestamp"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_card(record: dict, threshold: float) -> None:
    is_low_confidence = record["confidence"] < threshold
    badge_class = "badge-helpful" if record["prediction"] == 1 else "badge-not-helpful"
    badge_icon = "&#9989;" if record["prediction"] == 1 else "&#10060;"
    safe_review = html.escape(record["review_text"])

    st.markdown(
        f"""
        <div class="history-card">
            <span class="badge {badge_class}">{badge_icon} {record["label"]}</span>
            <div style="margin-top:0.4rem;">{safe_review}</div>
            <div class="meta">
                Confidence: {record["confidence"] * 100:.2f}% |
                Words: {record["word_count"]} |
                Low Confidence: {"Yes" if is_low_confidence else "No"} |
                {record["timestamp"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_history(
    history: list[dict],
    helpful_only: bool,
    low_conf_only: bool,
    long_only: bool,
    threshold: float,
) -> list[dict]:
    filtered = history
    if helpful_only:
        filtered = [item for item in filtered if item["prediction"] == 1]
    if low_conf_only:
        filtered = [item for item in filtered if item["confidence"] < threshold]
    if long_only:
        filtered = [
            item for item in filtered if item["word_count"] > LONG_REVIEW_MIN_WORDS
        ]
    return filtered


inject_styles()
init_session_state()

model, vectorizer, load_error = load_model()

with st.sidebar:
    st.header("Model Information")
    st.markdown(
        """
        **Model Type:** Logistic Regression  
        **Artifacts:** `helpfulness_model.pkl`, `helpfulness_vectorizer.pkl`  
        **Inference Policy:** Rating/Score is not used in prediction.
        """
    )

    st.divider()
    st.header("Dataset Info")
    st.markdown(
        """
        **Training Corpus:** Amazon Food Reviews  
        **Label Rule:** Helpful ratio >= 0.60
        """
    )

    st.divider()
    st.header("Features Used")
    st.markdown(
        f"""
        - TF-IDF Features: {EXPECTED_TFIDF_COUNT}
        - Review Length: 1
        - Total Features: {DATASET_STATS["feature_count"]}
        """
    )

    st.divider()
    st.header("Confidence Threshold")
    confidence_threshold = st.slider(
        "Minimum confidence for reliable prediction",
        min_value=0.50,
        max_value=1.00,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.01,
    )

    st.divider()
    st.header("Dataset Statistics")
    st.markdown(
        f"""
        - Raw Rows: {DATASET_STATS["raw_rows"]:,}
        - Valid Rows: {DATASET_STATS["valid_rows"]:,}
        - Training Sample: {DATASET_STATS["training_sample"]:,}
        - Helpful: {DATASET_STATS["helpful_pct"]:.2f}%
        - Not Helpful: {DATASET_STATS["not_helpful_pct"]:.2f}%
        """
    )

    st.divider()
    st.header("Filters")
    filter_helpful = st.checkbox("Show only Helpful", value=False)
    filter_low_conf = st.checkbox("Show Low Confidence", value=False)
    filter_long_reviews = st.checkbox(
        f"Show Long Reviews (> {LONG_REVIEW_MIN_WORDS} words)", value=False
    )

if load_error:
    st.error(load_error)
    st.stop()

st.markdown(
    """
    <div class="app-header">
        <h2 style="margin:0;">Review Intelligence Dashboard</h2>
        <div class="subtle-text">AI-powered review helpfulness system</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.subheader("Input Panel")
review_text = st.text_area(
    "Enter customer review",
    height=140,
    placeholder=(
        "Example: This review explains product quality, durability, value, "
        "and usage outcomes in detail."
    ),
)
analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

if analyze_clicked:
    if not review_text.strip():
        st.warning("Please enter a review before running analysis.")
    else:
        try:
            prediction_record = predict_helpfulness(review_text, confidence_threshold)
            if prediction_record is None:
                st.warning("The review became empty after preprocessing.")
            else:
                st.session_state["last_prediction"] = prediction_record
                st.session_state["prediction_history"].append(prediction_record)
        except Exception as exc:
            st.error(f"Prediction error: {exc}")

latest = st.session_state.get("last_prediction")

if latest is not None:
    st.divider()
    st.subheader("Analytics Panel")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Prediction", latest["label"])
    metric_col2.metric("Confidence", f"{latest['confidence'] * 100:.2f}%")
    metric_col3.metric("Word Count", f"{latest['word_count']}")

    st.divider()
    st.subheader("Prediction Card")
    render_prediction_card(latest, confidence_threshold)

    if latest["confidence"] < confidence_threshold:
        st.warning(
            "Low Confidence warning: prediction confidence is below your selected "
            "threshold."
        )

    st.divider()
    st.subheader("Business Insights Panel")
    if latest["prediction"] == 1:
        st.markdown(
            """
            <div class="insight-box">
                <strong>Helpful Review Signals</strong><br>
                - Detailed / informative review<br>
                - Good for top ranking
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="insight-box">
                <strong>Not Helpful Review Signals</strong><br>
                - Too short / generic<br>
                - Suggest adding more details
            </div>
            """,
            unsafe_allow_html=True,
        )

    if latest["confidence"] < confidence_threshold:
        st.info(
            "Caution: this result is below threshold confidence and should be "
            "manually reviewed before high-impact decisions."
        )

st.divider()
st.subheader("History Panel")

filtered_history = filter_history(
    st.session_state["prediction_history"],
    helpful_only=filter_helpful,
    low_conf_only=filter_low_conf,
    long_only=filter_long_reviews,
    threshold=confidence_threshold,
)

if not filtered_history:
    st.info("No prediction records match the active filters.")
else:
    for item in reversed(filtered_history):
        render_history_card(item, confidence_threshold)
