from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import joblib

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset(path):
    df = pd.read_csv(path)
    required = {"message", "label", "campaign_group", "channel"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["message", "label", "campaign_group"]).copy()
    df["message"] = df["message"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0,1])]
    df["clean_message"] = df["message"].map(clean_text)
    df = df[df["clean_message"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean_message"]).reset_index(drop=True)
    return df

def rule_score(message):
    text = str(message).lower()
    cues = [
        (r"\b(urgent|immediately|act now|final warning|today only)\b", 17, "Urgency or pressure language"),
        (r"\b(verify|confirm|update|validate)\b.{0,35}\b(account|identity|details|password|payment)\b", 22, "Request to verify sensitive information"),
        (r"\b(blocked|suspended|locked|disabled|terminated)\b", 16, "Threat of account restriction"),
        (r"\b(prize|winner|won|reward|gift card|claim your)\b", 17, "Unexpected reward or prize wording"),
        (r"\b(password|otp|one[- ]time code|pin|bank details)\b", 20, "Credential or financial information mentioned"),
        (r"\b(click|open|visit|follow)\b.{0,35}\b(link|url|here|below)\b", 12, "Prompt to follow a link"),
        (r"\b(invoice|payment|refund)\b.{0,45}\b(pending|failed|overdue|release)\b", 12, "Payment-related pressure"),
    ]
    score, reasons = 8, []
    for pattern, weight, reason in cues:
        if re.search(pattern, text, flags=re.I):
            score += weight
            reasons.append(reason)
    score = min(score, 98)
    if not reasons:
        reasons.append("No common high-pressure cue matched; this does not prove the message is safe")
    return score, reasons

def _make_models():
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=12000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=12000, sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=.35))
        ])
    }

def train_models(df, model_dir, report_dir):
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["campaign_group"]))
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    # Ensure both classes are present in both partitions for useful metrics.
    if train.label.nunique() < 2 or test.label.nunique() < 2:
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=.25, random_state=42, stratify=df.label)
    X_train, y_train = train.clean_message, train.label
    X_test, y_test = test.clean_message, test.label
    models = _make_models()
    rows, predictions = [], []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
        rows.append({
            "model": name, "accuracy": accuracy_score(y_test,pred),
            "precision": precision_score(y_test,pred,zero_division=0),
            "recall": recall_score(y_test,pred,zero_division=0),
            "f1": f1_score(y_test,pred,zero_division=0),
            "false_positive_rate": fp/(fp+tn) if fp+tn else 0,
            "test_rows": len(test), "train_campaigns": train.campaign_group.nunique(),
            "test_campaigns": test.campaign_group.nunique()
        })
        for idx, actual, predicted in zip(test.index, y_test, pred):
            predictions.append({"row_id":int(idx),"model":name,"actual_label":int(actual),"predicted_label":int(predicted),"message_preview":str(df.loc[idx,"message"])[:140]})
    # Rules baseline
    rule_pred = [1 if rule_score(x)[0] >= 48 else 0 for x in test.message]
    tn, fp, fn, tp = confusion_matrix(y_test, rule_pred, labels=[0,1]).ravel()
    rows.append({
        "model":"Rules Baseline", "accuracy":accuracy_score(y_test,rule_pred),
        "precision":precision_score(y_test,rule_pred,zero_division=0),
        "recall":recall_score(y_test,rule_pred,zero_division=0),
        "f1":f1_score(y_test,rule_pred,zero_division=0),
        "false_positive_rate":fp/(fp+tn) if fp+tn else 0,
        "test_rows":len(test), "train_campaigns":train.campaign_group.nunique(),
        "test_campaigns":test.campaign_group.nunique()
    })
    evaluation = pd.DataFrame(rows)
    joblib.dump(models, model_dir/"phishshift_models.joblib")
    evaluation.to_csv(report_dir/"model_results.csv", index=False)
    pd.DataFrame(predictions).to_csv(report_dir/"holdout_predictions.csv", index=False)
    return models, evaluation, test

def analyze_message(message, models):
    if not str(message).strip():
        return {"risk":"Uncertain","score":0.0,"model":"No input","reasons":["Enter a message to analyze"],"note":"No message was supplied."}
    clean = clean_text(message)
    scores = []
    for name, model in models.items():
        try:
            scores.append((name, float(model.predict_proba([clean])[0][1] * 100)))
        except Exception:
            pred = int(model.predict([clean])[0])
            scores.append((name, 100.0 if pred else 0.0))
    avg = float(np.mean([x[1] for x in scores]))
    rule, reasons = rule_score(message)
    # Combine model and transparent rule signal for a triage score.
    score = min(99.0, max(1.0, avg * .78 + rule * .22))
    spread = max(x[1] for x in scores) - min(x[1] for x in scores)
    if spread > 48 and 30 < score < 70:
        risk = "Uncertain"
    elif score >= 70:
        risk = "High"
    elif score >= 43:
        risk = "Medium"
    else:
        risk = "Low"
    if rule >= 48:
        reasons = reasons[:5]
    else:
        reasons = ["No strong rule-based warning pattern detected"] + reasons[:2]
    note = "Verify sensitive requests through a known official channel. This score is not a calibrated probability."
    return {"risk":risk,"score":score,"model":"LR + Multinomial NB + rules","reasons":reasons,"note":note}
