from pathlib import Path
from datetime import datetime
import html
import sys

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.pipeline import load_dataset, train_models, analyze_message

st.set_page_config(page_title="PhishShift | Active Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# Liquid-glass visual system: translucent surfaces, soft glows, restrained cyan/violet accents.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#070b14;--glass:rgba(18,27,46,.62);--line:rgba(183,205,255,.15);--text:#edf3ff;--muted:#96a8c7;--cyan:#66f0dc;--blue:#8bb7ff;--violet:#b9a2ff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 10% 0%,rgba(36,86,131,.38),transparent 35%),radial-gradient(ellipse at 90% 10%,rgba(92,57,151,.25),transparent 32%),linear-gradient(135deg,#070b14,#0b1220 50%,#080c17);color:var(--text)}
[data-testid="stHeader"]{background:rgba(7,11,20,.4)}
[data-testid="stSidebar"]{background:rgba(10,17,31,.84);backdrop-filter:blur(22px);border-right:1px solid var(--line)}
[data-testid="stSidebar"] *{color:var(--text)}
.block-container{max-width:1500px;padding-top:1.6rem;padding-bottom:3rem}
#MainMenu,footer{visibility:hidden}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.5px}
.hero{position:relative;overflow:hidden;border:1px solid rgba(166,199,255,.22);border-radius:28px;padding:30px 32px;background:linear-gradient(120deg,rgba(23,43,70,.76),rgba(19,28,50,.53) 55%,rgba(40,43,83,.48));backdrop-filter:blur(24px);box-shadow:0 20px 70px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.08);margin-bottom:22px}
.hero:before{content:"";position:absolute;width:280px;height:280px;right:-65px;top:-150px;border-radius:50%;border:1px solid rgba(102,240,220,.3);box-shadow:0 0 0 30px rgba(102,240,220,.035),0 0 0 65px rgba(139,183,255,.025)}
.eyebrow{font-size:.7rem;letter-spacing:2.2px;text-transform:uppercase;font-weight:800;color:var(--cyan)}
.hero h1{font-size:2.45rem;font-weight:800;color:#f5f8ff;margin:.4rem 0 .5rem}
.hero p{color:#b2c2dd;max-width:780px;margin:0;font-size:1rem}
.pill{display:inline-flex;align-items:center;gap:8px;border:1px solid rgba(102,240,220,.25);border-radius:999px;padding:8px 12px;background:rgba(102,240,220,.07);color:#a4fff0;font-size:.78rem;font-weight:700}
.dot{height:7px;width:7px;background:#66f0dc;border-radius:50%;box-shadow:0 0 12px #66f0dc}
.glass{background:linear-gradient(145deg,rgba(24,36,59,.68),rgba(14,22,39,.58));border:1px solid var(--line);border-radius:20px;padding:20px;backdrop-filter:blur(22px);box-shadow:0 12px 36px rgba(0,0,0,.16),inset 0 1px 0 rgba(255,255,255,.045);margin-bottom:15px}
.metric{min-height:120px}
.metric-label{font-size:.72rem;letter-spacing:1.2px;color:#9eafcd;text-transform:uppercase;font-weight:700}
.metric-value{font-family:'Manrope',sans-serif;font-size:2rem;font-weight:800;color:#f3f7ff;margin-top:8px;line-height:1.15}
.metric-foot{font-size:.77rem;color:#8fa2c3;margin-top:8px}
.section-title{font-family:'Manrope',sans-serif;font-size:1.15rem;font-weight:800;color:#f0f5ff;margin:.2rem 0}
.section-sub{font-size:.84rem;color:var(--muted);margin-bottom:14px}
.stButton>button{border-radius:12px;border:1px solid rgba(139,183,255,.3);background:rgba(28,47,76,.75);color:#edf3ff;font-weight:700;min-height:2.7rem;transition:all .2s}
.stButton>button[kind="primary"]{background:linear-gradient(105deg,#66f0dc,#8bb7ff);color:#07121b;border:0;box-shadow:0 6px 22px rgba(102,240,220,.14)}
.stButton>button:hover{border-color:var(--cyan);transform:translateY(-1px)}
.stTextArea textarea,.stTextInput input{background:rgba(7,14,27,.75)!important;color:#edf3ff!important;border:1px solid rgba(183,205,255,.2)!important;border-radius:13px!important}
[data-baseweb="select"]>div{background:rgba(7,14,27,.8);border-color:rgba(183,205,255,.2);border-radius:12px}
.stTabs [data-baseweb="tab-list"]{gap:7px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;background:rgba(24,36,59,.45);color:#aab9d3;padding:10px 15px}
.stTabs [aria-selected="true"]{background:rgba(102,240,220,.1)!important;color:#a4fff0!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}
[data-testid="stMetric"]{background:rgba(24,36,59,.5);border:1px solid var(--line);padding:14px;border-radius:14px}
[data-testid="stMetricLabel"]{color:#9eafcd}
[data-testid="stMetricValue"]{color:#f3f7ff;font-family:'Manrope',sans-serif}
hr{border-color:var(--line)!important}
.note{font-size:.78rem;color:#8fa2c3;line-height:1.6}
.risk-high{color:#ff8eaa}.risk-medium{color:#ffd28a}.risk-low{color:#78efc2}.risk-uncertain{color:#b8c9e8}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='font-size:2rem;color:#66f0dc'>◈</div><div style='font-family:Manrope;font-weight:800;font-size:1.35rem'>PhishShift</div><div style='font-size:.72rem;color:#96a8c7;letter-spacing:1.5px'>ACTIVE INTELLIGENCE</div>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Workspace", ["Active Dashboard", "Live Message Scanner", "Model Analytics", "Project Notes"], label_visibility="visible")
    st.divider()
    st.markdown("<div class='note'>● LOCAL WORKSPACE<br>Messages are analyzed locally.<br>No links are opened or fetched.</div>", unsafe_allow_html=True)
    st.markdown("<div class='note' style='margin-top:28px'>ML-07 · CYBERSECURITY NLP<br>EDUCATIONAL PROTOTYPE</div>", unsafe_allow_html=True)

with st.spinner("Loading local dataset and training models on first run…"):
    df = load_dataset(ROOT / "data" / "raw" / "messages.csv")
    models, evaluation, holdout = train_models(df, ROOT / "models", ROOT / "reports")

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

st.markdown("""
<div class="hero">
 <div class="eyebrow">ML-07 / CYBERSECURITY NLP</div>
 <h1>Message Risk Intelligence</h1>
 <p>An active workspace for message triage, model monitoring and campaign-aware evaluation — presented in a premium liquid-glass interface.</p>
</div>""", unsafe_allow_html=True)
st.markdown("<span class='pill'><span class='dot'></span> SYSTEM ONLINE &nbsp;·&nbsp; LOCAL MODEL READY</span>", unsafe_allow_html=True)
st.write("")

n = len(df)
phish = int(df.label.sum())
legit = n - phish
scans = len(st.session_state.scan_history)
best = evaluation.sort_values("f1", ascending=False).iloc[0] if len(evaluation) else None
best_f1 = float(best.f1) if best is not None else 0.0
best_model = str(best.model) if best is not None else "N/A"

if page == "Active Dashboard":
    st.markdown("<div class='section-title'>Live operational overview</div><div class='section-sub'>Session scan activity updates as you analyze messages. Model metrics come from the campaign-held-out test set.</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [
        ("DATASET MESSAGES", f"{n:,}", "Synthetic examples in local corpus"),
        ("PHISHING / LEGITIMATE", f"{phish:,} / {legit:,}", "Labeled training examples"),
        ("SESSION SCANS", f"{scans:,}", "Messages scanned in this browser session"),
        ("BEST HOLDOUT F1", f"{best_f1:.1%}", best_model),
    ]
    for col,(label,value,foot) in zip(cols,cards):
        col.markdown(f"<div class='glass metric'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-foot'>{html.escape(foot)}</div></div>",unsafe_allow_html=True)

    c1,c2 = st.columns([1.15,1], gap="large")
    with c1:
        st.markdown("<div class='glass'><div class='section-title'>Model performance snapshot</div><div class='section-sub'>Percentage scores on campaign-held-out samples.</div>",unsafe_allow_html=True)
        if len(evaluation):
            chart_df = evaluation.melt(id_vars="model", value_vars=["precision","recall","f1","false_positive_rate"], var_name="metric", value_name="score")
            chart_df["metric"] = chart_df.metric.map({"precision":"Precision","recall":"Recall","f1":"F1 score","false_positive_rate":"False-positive rate"})
            fig = px.bar(chart_df, x="model", y="score", color="metric", barmode="group", range_y=[0,1.08],
                         color_discrete_sequence=["#8bb7ff","#66f0dc","#b9a2ff","#ffbd82"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#dce7fb",
                              legend_title_text="",margin=dict(l=10,r=10,t=20,b=10),yaxis=dict(tickformat=".0%",gridcolor="rgba(183,205,255,.12)",title="Score"),
                              xaxis=dict(title="",tickfont_size=11))
            st.plotly_chart(fig,use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass'><div class='section-title'>Dataset composition</div><div class='section-sub'>Current label distribution.</div>",unsafe_allow_html=True)
        donut = pd.DataFrame({"Label":["Legitimate","Phishing"],"Count":[legit,phish]})
        fig = px.pie(donut, names="Label", values="Count", hole=.7, color="Label",
                     color_discrete_map={"Legitimate":"#8bb7ff","Phishing":"#66f0dc"})
        fig.update_traces(textinfo="percent+label",textfont_color="#edf3ff",marker_line_color="#111b2d",marker_line_width=3)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#dce7fb",showlegend=False,
                          margin=dict(l=5,r=5,t=5,b=5),annotations=[dict(text=f"{n}<br><sup>messages</sup>",x=.5,y=.5,showarrow=False,font=dict(size=20,color="#edf3ff"))])
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Active scan table</div><div class='section-sub'>New scans appear here immediately. This history lasts for the current Streamlit session.</div>",unsafe_allow_html=True)
    if st.session_state.scan_history:
        hist = pd.DataFrame(st.session_state.scan_history)
        show = hist[["time","risk","score","model","preview"]].rename(columns={"time":"Time","risk":"Risk level","score":"Risk score (%)","model":"Model","preview":"Message preview"})
        st.dataframe(show, use_container_width=True, hide_index=True, column_config={"Risk score (%)":st.column_config.ProgressColumn("Risk score (%)",min_value=0,max_value=100,format="%.1f%%")})
        csv_bytes = hist.to_csv(index=False).encode("utf-8")
        st.download_button("Export active scan table (CSV)",csv_bytes,"phishshift_active_scans.csv","text/csv")
    else:
        st.info("No messages scanned in this session yet. Open **Live Message Scanner**, analyze a sample, then return here to see the active table.")
    st.markdown("<div class='note'>Prototype notice: the bundled corpus is synthetic. Percentages are model scores, not calibrated probabilities or a guarantee of safety.</div>",unsafe_allow_html=True)

elif page == "Live Message Scanner":
    st.markdown("<div class='section-title'>Live message scanner</div><div class='section-sub'>Analyze a message and append the result to the active dashboard table.</div>",unsafe_allow_html=True)
    left,right = st.columns([1.2,.8],gap="large")
    sample_options = {
        "Suspicious account alert":"URGENT: Your bank account will be blocked today. Verify your account immediately to avoid suspension. Link: hxxps://secure-login-example[.]com",
        "Legitimate college reminder":"Hi, your college assignment submission is scheduled for tomorrow. Please submit your work before 5 PM.",
        "Legitimate password reset":"We received a request to reset your password. If you made this request, follow the instructions on the official website. If not, you can ignore this message.",
        "Urgent payment review":"Action required: Your payment is pending. Please review your transaction status through the official application.",
        "Unfamiliar wording":"Acct access review needed. Pls verify details soon.",
    }
    with left:
        st.markdown("<div class='glass'>",unsafe_allow_html=True)
        choice = st.selectbox("Quick sample",["Choose a sample…"]+list(sample_options))
        if st.button("Load sample",use_container_width=True) and choice in sample_options:
            st.session_state.scan_text = sample_options[choice]
        message = st.text_area("Message text",key="scan_text",height=220,placeholder="Paste a harmless sample email or SMS here…")
        if st.button("Analyze message",type="primary",use_container_width=True):
            result = analyze_message(message,models)
            if not message.strip():
                st.warning("Enter a message before scanning.")
            else:
                record = {"time":datetime.now().strftime("%H:%M:%S"),"risk":result["risk"],"score":round(result["score"],1),"model":result["model"],"preview":message.strip()[:120]}
                st.session_state.scan_history.insert(0,record)
                st.session_state.last_result = result
                st.rerun()
        st.markdown("<div class='note'>Do not paste real credentials, private messages or sensitive personal information. The scanner never visits URLs.</div></div>",unsafe_allow_html=True)
    with right:
        st.markdown("<div class='glass'><div class='section-title'>Risk interpretation</div><p><span class='risk-high'><b>HIGH</b></span> · Strong suspicious-language signal</p><p><span class='risk-medium'><b>MEDIUM</b></span> · Review context and sender independently</p><p><span class='risk-low'><b>LOW</b></span> · No strong signal found; not proof of safety</p><p><span class='risk-uncertain'><b>UNCERTAIN</b></span> · Borderline or model disagreement</p><hr><div class='note'>A model score is not a verified probability. Confirm sensitive requests using a known official channel.</div></div>",unsafe_allow_html=True)
    if "last_result" in st.session_state:
        r = st.session_state.last_result
        css = {"High":"risk-high","Medium":"risk-medium","Low":"risk-low","Uncertain":"risk-uncertain"}.get(r["risk"],"risk-uncertain")
        st.markdown(f"<div class='glass'><div class='eyebrow'>LATEST SCAN RESULT</div><div class='{css}' style='font-family:Manrope;font-size:2rem;font-weight:800'>{html.escape(r['risk'])} risk</div><div class='metric-value' style='font-size:1.7rem'>{r['score']:.1f}% <span style='font-size:.85rem;color:#96a8c7'>model score</span></div><p class='note'>Model: {html.escape(r['model'])}</p></div>",unsafe_allow_html=True)
        st.markdown("**Signals to review**")
        for reason in r["reasons"]:
            st.markdown(f"- {reason}")
        st.info(r["note"])

elif page == "Model Analytics":
    st.markdown("<div class='section-title'>Model analytics</div><div class='section-sub'>Rules baseline vs TF-IDF + Logistic Regression vs Multinomial Naive Bayes.</div>",unsafe_allow_html=True)
    st.dataframe(evaluation.style.format({c:"{:.1%}" for c in ["accuracy","precision","recall","f1","false_positive_rate"]}),use_container_width=True,hide_index=True)
    st.markdown("<div class='glass'><div class='section-title'>Metric definitions</div><p><b>Precision</b> — Of messages predicted as phishing, how many were phishing in the test set.</p><p><b>Recall</b> — Of phishing messages in the test set, how many were detected.</p><p><b>False-positive rate</b> — Share of legitimate messages incorrectly flagged as phishing.</p><p><b>F1 score</b> — Harmonic mean of precision and recall.</p></div>",unsafe_allow_html=True)
    if (ROOT/"reports"/"model_results.csv").exists():
        st.download_button("Download model results CSV",(ROOT/"reports"/"model_results.csv").read_bytes(),"phishshift_model_results.csv","text/csv")

else:
    st.markdown("<div class='section-title'>Project notes</div><div class='section-sub'>Implementation and responsible-use notes for your presentation and report.</div>",unsafe_allow_html=True)
    st.markdown("""
    <div class='glass'>
    <h4>01 · Data preparation</h4><p>Uses a local CSV corpus, validates columns, normalizes text and removes exact duplicates.</p>
    <h4>02 · Models</h4><p>Compares a rules baseline, TF-IDF + Logistic Regression, and TF-IDF + Multinomial Naive Bayes.</p>
    <h4>03 · Evaluation</h4><p>Splits by campaign group so a campaign does not appear in both train and test sets.</p>
    <h4>04 · Active dashboard</h4><p>Every scan adds a timestamp, risk category, score, model and message preview to the current-session table.</p>
    <h4>05 · Limitations</h4><p>Dataset is synthetic; results are educational only. Add licensed real-world examples and unseen-campaign testing before making performance claims.</p>
    </div>
    """,unsafe_allow_html=True)

st.divider()
st.markdown("<div class='note'>PHISHSHIFT · ML-07 · LIQUID GLASS EDITION · Educational triage prototype only.</div>",unsafe_allow_html=True)

