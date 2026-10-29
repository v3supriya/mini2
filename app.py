
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# PHISHSHIFT - Standalone phishing-message risk dashboard
# Educational demo using synthetic messages only.
# ---------------------------------------------------------

st.set_page_config(
    page_title="PhishShift | Message Security",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f 0%, #101b31 100%);
    color: #eef4ff;
}
[data-testid="stSidebar"] {
    background: #0b1425;
}
h1, h2, h3 {
    color: #eaf2ff;
}
div[data-testid="stMetric"] {
    background: rgba(40, 60, 95, 0.35);
    border: 1px solid #293b59;
    padding: 16px;
    border-radius: 14px;
}
.stButton button, .stDownloadButton button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. Create a synthetic demonstration dataset.
#    These are generated examples, not real user messages.
# ---------------------------------------------------------

@st.cache_data
def create_dataset():
    legitimate_templates = [
        "Your appointment is confirmed for {day} at {time}.",
        "The meeting has been moved to {day}. Please check your calendar.",
        "Your order has been shipped. Track it in the official app.",
        "The class assignment is available on the college portal.",
        "Your payment receipt is ready in your account.",
        "Reminder: your library book is due on {day}.",
        "The project review starts at {time} in the lab.",
        "Your ticket confirmation is available in the official app.",
        "Please attend the scheduled team meeting at {time}.",
        "Your monthly statement is ready. Sign in through the official app.",
        "The exam timetable has been published on the college website.",
        "Your subscription renewal reminder is available in your account.",
        "Your support request has been received. Reference number {number}.",
        "The delivery is scheduled for {day}.",
        "Please review the shared document before the meeting.",
    ]

    suspicious_templates = [
        "Urgent: verify your account immediately to avoid suspension.",
        "Your account is locked. Confirm your password to restore access.",
        "You have won a prize. Submit your personal details to claim it.",
        "Your payment failed. Use this unfamiliar link to update your details.",
        "Immediate action required: your account will be disabled today.",
        "Confirm your bank details now to prevent account restrictions.",
        "Your parcel is on hold. Pay an unexpected fee to continue delivery.",
        "Security alert: provide your login code to verify your identity.",
        "You are eligible for a refund. Enter your card details to receive it.",
        "Your account has unusual activity. Verify your credentials now.",
        "Claim your reward today by submitting confidential information.",
        "Your access expires soon. Sign in through this unverified link.",
        "A problem was found with your account. Share your one-time code.",
        "You have an unpaid fee. Make an urgent payment to avoid a penalty.",
        "Your account needs verification. Provide your password immediately.",
    ]

    rows = []
    for i in range(240):
        template = legitimate_templates[i % len(legitimate_templates)]
        rows.append({
            "message": template.format(
                day=["Monday", "Tuesday", "Friday", "tomorrow"][i % 4],
                time=["9:00 AM", "2:30 PM", "4:00 PM"][i % 3],
                number=1000 + i,
            ),
            "label": 0,
            "channel": ["Email", "SMS", "Chat"][i % 3],
        })

    for i in range(240):
        template = suspicious_templates[i % len(suspicious_templates)]
        rows.append({
            "message": template,
            "label": 1,
            "channel": ["Email", "SMS", "Chat"][i % 3],
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------
# 2. Train the text classification models.
# ---------------------------------------------------------

@st.cache_resource
def train_models():
    df = create_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            )),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=5000,
            )),
            ("model", MultinomialNB()),
        ]),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        metrics[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        }

    return models, metrics


# ---------------------------------------------------------
# 3. Rule-based baseline.
#    These are simple indicators, not proof of phishing.
# ---------------------------------------------------------

RISK_PATTERNS = [
    r"\bverify your account\b",
    r"\bconfirm your password\b",
    r"\baccount is locked\b",
    r"\bshare your (?:one-time|login) code\b",
    r"\burgent\b",
    r"\bclaim your (?:prize|reward)\b",
    r"\bprovide your bank details\b",
    r"\bavoid account suspension\b",
    r"\benter your card details\b",
    r"\bimmediate action required\b",
]

def rule_score(message):
    text = message.lower()
    matches = [p for p in RISK_PATTERNS if re.search(p, text)]
    score = min(len(matches) / 3, 1.0)
    return score, len(matches)


# ---------------------------------------------------------
# 4. Analyze one message using all three methods.
# ---------------------------------------------------------

def analyze_message(message, models):
    cleaned = message.strip()

    if not cleaned:
        return None

    rule_probability, rule_matches = rule_score(cleaned)

    lr_probability = float(
        models["Logistic Regression"].predict_proba([cleaned])[0][1]
    )
    nb_probability = float(
        models["Naive Bayes"].predict_proba([cleaned])[0][1]
    )

    # Equal-weight demo ensemble. Scores are model estimates,
    # not guaranteed real-world probabilities.
    combined = (
        0.40 * lr_probability
        + 0.35 * nb_probability
        + 0.25 * rule_probability
    )

    if len(cleaned.split()) < 3:
        risk = "Uncertain"
    elif combined >= 0.70:
        risk = "High"
    elif combined >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "message": cleaned,
        "risk": risk,
        "combined_score": combined,
        "rule_score": rule_probability,
        "rule_matches": rule_matches,
        "lr_score": lr_probability,
        "nb_score": nb_probability,
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------
# 5. Dashboard layout.
# ---------------------------------------------------------

models, model_metrics = train_models()
dataset = create_dataset()

st.title("🛡️ PhishShift")
st.caption(
    "Message Threat Detection • Rule Baseline • Machine Learning"
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a section",
        ["Overview", "Scan Message", "Dataset", "Model Evaluation"],
    )
    st.divider()
    st.caption("Demo uses synthetic data. Verify suspicious messages "
               "through official channels.")

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if page == "Overview":
    st.subheader("Security Overview")

    total_scans = len(st.session_state.scan_history)
    high_scans = sum(
        row["risk"] == "High" for row in st.session_state.scan_history
    )
    medium_scans = sum(
        row["risk"] == "Medium" for row in st.session_state.scan_history
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Examples", len(dataset))
    c2.metric("Messages Scanned", total_scans)
    c3.metric("High Risk", high_scans)
    c4.metric("Medium Risk", medium_scans)

    st.markdown("### Recent Scan Results")
    if st.session_state.scan_history:
        history = pd.DataFrame(st.session_state.scan_history)
        st.dataframe(
            history[[
                "scanned_at", "risk", "combined_score", "message"
            ]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No messages scanned yet. Open Scan Message to begin.")

    st.markdown("### Detection workflow")
    st.write(
        "Message → Rule indicators → TF-IDF text features → "
        "Logistic Regression + Naive Bayes → Combined risk estimate"
    )

elif page == "Scan Message":
    st.subheader("Scan a Message")
    st.write("Paste a message to estimate its risk.")

    with st.form("scan_form"):
        message = st.text_area(
            "Message text",
            placeholder="Paste the message you want to check...",
            height=150,
        )
        submitted = st.form_submit_button("Analyze Message")

    if submitted:
        result = analyze_message(message, models)
        if result is None:
            st.warning("Please enter a message first.")
        else:
            st.session_state.scan_history.insert(0, result)

            if result["risk"] == "High":
                st.error("HIGH RISK — review this message carefully.")
            elif result["risk"] == "Medium":
                st.warning("MEDIUM RISK — verify using an official channel.")
            elif result["risk"] == "Uncertain":
                st.info("UNCERTAIN — the message is too short to assess well.")
            else:
                st.success("LOW RISK ESTIMATE — still check the sender.")

            a, b, c = st.columns(3)
            a.metric("Risk Level", result["risk"])
            b.metric("Combined Score", f"{result['combined_score']:.1%}")
            c.metric("Rule Indicators", result["rule_matches"])

            st.markdown("### Model scores")
            score_df = pd.DataFrame({
                "Method": [
                    "Rule baseline",
                    "Logistic Regression",
                    "Naive Bayes",
                ],
                "Estimated score": [
                    result["rule_score"],
                    result["lr_score"],
                    result["nb_score"],
                ],
            })
            st.bar_chart(score_df.set_index("Method"))

            st.caption(
                "Scores are estimates from a synthetic demonstration dataset. "
                "They are not proof that a message is safe or malicious."
            )

    st.markdown("### Scan history")
    if st.session_state.scan_history:
        history = pd.DataFrame(st.session_state.scan_history)
        st.dataframe(history, use_container_width=True, hide_index=True)
        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scan history (CSV)",
            data=csv_data,
            file_name="phishshift_scan_history.csv",
            mime="text/csv",
        )

elif page == "Dataset":
    st.subheader("Synthetic Dataset")
    st.write(
        "This demo creates 480 generated examples: 240 legitimate and "
        "240 suspicious. These are not real messages or a validated dataset."
    )
    st.dataframe(dataset, use_container_width=True, hide_index=True)

    csv_data = dataset.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download dataset (CSV)",
        data=csv_data,
        file_name="phishshift_synthetic_dataset.csv",
        mime="text/csv",
    )

elif page == "Model Evaluation":
    st.subheader("Model Evaluation")
    st.write(
        "Metrics below are measured on a held-out split of the generated "
        "synthetic examples. They should not be treated as real-world "
        "performance estimates."
    )

    metrics_df = pd.DataFrame(model_metrics).T
    st.dataframe(
        metrics_df.style.format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown("**Metric definitions**")
    st.write("- **Precision:** Of messages flagged, how many were positive.")
    st.write("- **Recall:** Of positive messages, how many were detected.")
    st.write("- **FPR:** Of legitimate messages, how many were incorrectly flagged.")
    st.write("- **Accuracy:** Fraction of all predictions that were correct.")

st.divider()
st.caption("PhishShift • Educational prototype • Not a production security filter")
