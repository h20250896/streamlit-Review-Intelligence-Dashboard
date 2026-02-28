import html
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix, hstack


MODEL_FILE = Path("helpfulness_model.pkl")
VECTORIZER_FILE = Path("helpfulness_vectorizer.pkl")
METADATA_FILE = Path("helpfulness_metadata.json")

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_DECISION_THRESHOLD = 0.50
LONG_REVIEW_MIN_WORDS = 20

DEFAULT_DATASET_STATS = {
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


def build_dataset_stats(metadata: dict[str, Any]) -> dict[str, Any]:
    stats = dict(DEFAULT_DATASET_STATS)
    dataset_meta = metadata.get("dataset", {})

    if dataset_meta:
        stats["raw_rows"] = dataset_meta.get("raw_rows", stats["raw_rows"])
        stats["valid_rows"] = dataset_meta.get(
            "valid_rows_post_filters", stats["valid_rows"]
        )
        stats["training_sample"] = dataset_meta.get(
            "training_rows", stats["training_sample"]
        )
        helpful_pct = dataset_meta.get("class_1_pct")
        not_helpful_pct = dataset_meta.get("class_0_pct")
        if helpful_pct is not None:
            stats["helpful_pct"] = float(helpful_pct)
        if not_helpful_pct is not None:
            stats["not_helpful_pct"] = float(not_helpful_pct)

    return stats


def extract_slice_gaps(fairness_report: dict[str, Any], slice_name: str) -> tuple[float, float]:
    for report in fairness_report.get("slice_reports", []):
        if report.get("slice_name") == slice_name:
            gap_1 = float(report.get("recall_gap_class_1", 0.0))
            gap_0 = float(report.get("recall_gap_class_0", 0.0))
            return gap_1, gap_0
    return 0.0, 0.0


def get_vectorizer_feature_count(vectorizer_obj: Any) -> int:
    if hasattr(vectorizer_obj, "vocabulary_") and vectorizer_obj.vocabulary_:
        return int(len(vectorizer_obj.vocabulary_))
    if hasattr(vectorizer_obj, "get_feature_names_out"):
        try:
            return int(len(vectorizer_obj.get_feature_names_out()))
        except Exception:
            pass
    raise ValueError(
        "Unable to determine vectorizer feature count from vocabulary_ "
        "or get_feature_names_out()."
    )


@st.cache_resource
def load_model() -> tuple[object | None, object | None, dict[str, Any], str | None]:
    try:
        if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
            return None, None, {}, (
                "Model artifacts not found. Expected files in app directory: "
                f"{MODEL_FILE.name}, {VECTORIZER_FILE.name}."
            )

        loaded_model = joblib.load(MODEL_FILE)
        loaded_vectorizer = joblib.load(VECTORIZER_FILE)

        metadata: dict[str, Any] = {}
        if METADATA_FILE.exists():
            metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

        try:
            vocab_size = get_vectorizer_feature_count(loaded_vectorizer)
        except ValueError as exc:
            return None, None, metadata, str(exc)

        model_features = getattr(loaded_model, "n_features_in_", None)
        if model_features is None:
            return None, None, metadata, "Model is missing n_features_in_ compatibility metadata."

        model_meta = metadata.get("model", {})
        numeric_features = model_meta.get("numeric_features")
        inferred_numeric_count = max(int(model_features) - int(vocab_size), 1)
        expected_numeric_count = (
            len(numeric_features)
            if isinstance(numeric_features, list) and numeric_features
            else inferred_numeric_count
        )

        if vocab_size + expected_numeric_count != model_features:
            return None, None, metadata, (
                "Feature mismatch detected. Expected vectorizer_vocab + "
                f"numeric_feature_count == model_features, got {vocab_size} + "
                f"{expected_numeric_count} != {model_features}."
            )

        return loaded_model, loaded_vectorizer, metadata, None
    except Exception as exc:
        return None, None, {}, f"Error loading model artifacts: {exc}"


def preprocess_text(text: str) -> str:
    cleaned = str(text).lower()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    cleaned = re.sub(r"\d+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def resolve_tfidf_feature_count(
    vectorizer_obj: Any,
    model_meta: dict[str, Any],
    model_feature_count: int,
) -> int:
    try:
        return get_vectorizer_feature_count(vectorizer_obj)
    except Exception:
        pass

    metadata_vocab_size = model_meta.get("vectorizer_vocab_size")
    if isinstance(metadata_vocab_size, (int, float)) and int(metadata_vocab_size) > 0:
        return int(metadata_vocab_size)

    metadata_numeric_features = model_meta.get("numeric_features")
    numeric_feature_count = (
        len(metadata_numeric_features)
        if isinstance(metadata_numeric_features, list) and metadata_numeric_features
        else 1
    )
    return max(int(model_feature_count) - int(numeric_feature_count), 1)


def compute_semantic_score(cleaned_text: str) -> float:
    tokens = cleaned_text.split()
    if not tokens:
        return 0.0

    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / token_count
    top_token_ratio = max(Counter(tokens).values()) / token_count
    alpha_char_ratio = sum(char.isalpha() for char in cleaned_text) / max(
        len(cleaned_text), 1
    )
    semantic_score = (
        0.45 * unique_ratio
        + 0.35 * (1.0 - top_token_ratio)
        + 0.20 * alpha_char_ratio
    )
    return float(np.clip(semantic_score, 0.0, 1.0))


def build_numeric_feature_vector(cleaned_text: str, review_length: int) -> csr_matrix:
    numeric_values: list[float] = []
    for feature_name in model_numeric_features:
        if feature_name == "review_length_log_scaled":
            log_min = float(model_length_scaler.get("log_min", 0.0))
            log_range = max(float(model_length_scaler.get("log_range", 1.0)), 1e-6)
            scaled = (float(np.log1p(review_length)) - log_min) / log_range
            numeric_values.append(float(np.clip(scaled, 0.0, 1.0)))
        elif feature_name == "semantic_score":
            numeric_values.append(compute_semantic_score(cleaned_text))
        elif feature_name == "review_length_raw":
            numeric_values.append(float(review_length))
        else:
            raise ValueError(f"Unsupported numeric feature in metadata: {feature_name}")

    return csr_matrix(np.array([numeric_values], dtype=float))


def predict_helpfulness(review_text: str, threshold: float) -> dict | None:
    cleaned_text = preprocess_text(review_text)
    if not cleaned_text:
        return None

    tfidf_feature = vectorizer.transform([cleaned_text])
    review_length = len(cleaned_text.split())
    numeric_feature = build_numeric_feature_vector(cleaned_text, review_length)
    features = hstack([tfidf_feature, numeric_feature], format="csr")

    if features.shape[1] != model_feature_count:
        raise ValueError(
            f"Unexpected feature width {features.shape[1]}; expected "
            f"{model_feature_count}."
        )

    probabilities = model.predict_proba(features)[0]
    helpful_probability = float(probabilities[1])
    prediction = int(helpful_probability >= decision_threshold)
    confidence = float(max(helpful_probability, 1.0 - helpful_probability))
    label = "Helpful" if prediction == 1 else "Not Helpful"
    semantic_score = compute_semantic_score(cleaned_text)

    return {
        "review_text": review_text,
        "cleaned_text": cleaned_text,
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "helpful_probability": helpful_probability,
        "word_count": review_length,
        "semantic_score": semantic_score,
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
                Helpful Probability: {record["helpful_probability"] * 100:.2f}% |
                Confidence: {record["confidence"] * 100:.2f}% |
                Word Count: {record["word_count"]} |
                Semantic Score: {record.get("semantic_score", 0.0):.3f} |
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
                Helpful Prob: {record["helpful_probability"] * 100:.2f}% |
                Confidence: {record["confidence"] * 100:.2f}% |
                Words: {record["word_count"]} |
                Semantic: {record.get("semantic_score", 0.0):.3f} |
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

model, vectorizer, metadata, load_error = load_model()

if load_error:
    st.error(load_error)
    st.stop()

model_meta = metadata.get("model", {})
config_meta = metadata.get("config", {})
runtime_meta = metadata.get("runtime", {})
fairness_meta = metadata.get("fairness", {})

model_feature_count = int(getattr(model, "n_features_in_", 0))
tfidf_feature_count = resolve_tfidf_feature_count(
    vectorizer,
    model_meta,
    model_feature_count,
)
dataset_stats = build_dataset_stats(metadata)
dataset_stats["feature_count"] = model_feature_count

selected_model = model_meta.get("selected_model", "Logistic Regression")
decision_threshold = float(model_meta.get("decision_threshold", DEFAULT_DECISION_THRESHOLD))
decision_threshold = min(max(decision_threshold, 0.0), 1.0)
inferred_numeric_count = max(model_feature_count - tfidf_feature_count, 1)
model_numeric_features = model_meta.get("numeric_features")
if not isinstance(model_numeric_features, list) or not model_numeric_features:
    if inferred_numeric_count == 2:
        model_numeric_features = ["review_length_log_scaled", "semantic_score"]
    else:
        model_numeric_features = ["review_length_raw"]
model_length_scaler = model_meta.get("length_scaler", {})
if not isinstance(model_length_scaler, dict):
    model_length_scaler = {}
label_rule_threshold = float(
    config_meta.get("helpfulness_threshold", 0.60)
)
bias_flag = bool(fairness_meta.get("bias_flag", False))
max_recall_gap = float(fairness_meta.get("max_recall_gap", 0.0))

validation_results = model_meta.get("validation_results", {})
selected_validation = validation_results.get(selected_model, {})
threshold_diagnostics = selected_validation.get("threshold_diagnostics", {})
threshold_macro_f1 = float(threshold_diagnostics.get("macro_f1", 0.0))
threshold_sentiment_gap = float(threshold_diagnostics.get("sentiment_recall_gap", 0.0))
threshold_fairness_ok = bool(threshold_diagnostics.get("fairness_compliant", False))

fairness_cfg = config_meta.get("fairness", {})
sentiment_gap_limit = float(fairness_cfg.get("sentiment_max_recall_gap", 0.12))
global_gap_limit = float(fairness_cfg.get("max_recall_gap", 0.10))

sentiment_gap_class_1, sentiment_gap_class_0 = extract_slice_gaps(
    fairness_meta,
    "sentiment_polarity",
)

with st.sidebar:
    st.header("Model Information")
    st.markdown(
        f"""
        **Model Type:** {selected_model}  
        **Artifacts:** `{MODEL_FILE.name}`, `{VECTORIZER_FILE.name}`  
        **Decision Threshold:** {decision_threshold:.2f}  
        **Threshold Macro F1:** {threshold_macro_f1:.4f}  
        **Threshold Sentiment Gap:** {threshold_sentiment_gap:.4f}  
        **Threshold Fairness Compliant:** {"Yes" if threshold_fairness_ok else "No"}  
        **Inference Policy:** Rating/Score is not used in prediction.
        """
    )

    if runtime_meta:
        st.markdown(
            f"""
            **Runtime:** Python {runtime_meta.get("python_version", "N/A")}  
            **Scikit-Learn:** {runtime_meta.get("sklearn_version", "N/A")}
            """
        )

    st.divider()
    st.header("Dataset Info")
    st.markdown(
        f"""
        **Training Corpus:** Amazon Food Reviews  
        **Label Rule:** Adjusted helpfulness >= {label_rule_threshold:.2f}
        """
    )

    st.divider()
    st.header("Features Used")
    st.markdown(
        f"""
        - TF-IDF Features: {tfidf_feature_count}
        - Numeric Features: {len(model_numeric_features)} ({", ".join(model_numeric_features)})
        - Total Features: {model_feature_count}
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
        - Raw Rows: {int(dataset_stats["raw_rows"]):,}
        - Valid Rows: {int(dataset_stats["valid_rows"]):,}
        - Training Sample: {int(dataset_stats["training_sample"]):,}
        - Helpful: {dataset_stats["helpful_pct"]:.2f}%
        - Not Helpful: {dataset_stats["not_helpful_pct"]:.2f}%
        """
    )

    st.divider()
    st.header("Bias Audit")
    st.markdown(
        f"""
        - Bias Flag: {"Yes" if bias_flag else "No"}
        - Max Recall Gap: {max_recall_gap:.4f}
        - Allowed Max Gap: {global_gap_limit:.2f}
        - Sentiment Gap (Class 1): {sentiment_gap_class_1:.4f}
        - Sentiment Gap (Class 0): {sentiment_gap_class_0:.4f}
        - Allowed Sentiment Gap: {sentiment_gap_limit:.2f}
        """
    )

    st.divider()
    st.header("Filters")
    filter_helpful = st.checkbox("Show only Helpful", value=False)
    filter_low_conf = st.checkbox("Show Low Confidence", value=False)
    filter_long_reviews = st.checkbox(
        f"Show Long Reviews (> {LONG_REVIEW_MIN_WORDS} words)", value=False
    )

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

