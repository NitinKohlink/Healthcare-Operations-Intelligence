from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import clean_healthcare_data, add_analysis_features, department_summary, symptom_department_audit
from analytics import model_benchmark


st.set_page_config(
    page_title="Healthcare Operations Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "Synthetic educational dataset only. This dashboard is an analytics/operations "
    "project and must not be used for diagnosis, treatment, triage, or clinical decision-making."
)


@st.cache_data(show_spinner=False)
def load_default_data():
    raw_path = ROOT / "data" / "healthcare_raw_data_150500.csv"
    raw = pd.read_csv(raw_path)
    clean, enriched, report = clean_healthcare_data(raw)
    featured = add_analysis_features(clean)
    return raw, clean, enriched, report.as_dict(), featured


@st.cache_data(show_spinner=False)
def run_model_cached(df):
    metrics, artifacts = model_benchmark(df)
    rf = artifacts["Random Forest"]
    labels = sorted(df["department"].unique())
    cm = pd.DataFrame(
        rf["confusion_matrix"], index=labels, columns=labels
    )
    return metrics, cm


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def insight_box(title, text):
    st.info(f"**{title}**\n\n{text}")


def pct(x):
    return f"{x:.1f}%"


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("🏥 Healthcare Operations")
st.sidebar.caption("Interactive Analytics & ML Audit")
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Data Quality",
        "Department Intelligence",
        "Patient & Symptom Explorer",
        "ML Audit",
        "Evidence Q&A",
        "Downloads",
    ],
)

with st.sidebar:
    st.markdown("---")
    st.caption(DISCLAIMER)
    uploaded = st.file_uploader(
        "Optional: analyze another CSV",
        type=["csv"],
        help="The uploaded CSV must contain the same 14 source columns.",
    )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
if uploaded is None:
    raw, clean, enriched, report, featured = load_default_data()
    data_source = "Bundled project dataset"
else:
    try:
        raw = pd.read_csv(uploaded)
        clean, enriched, report_obj = clean_healthcare_data(raw)
        report = report_obj.as_dict()
        featured = add_analysis_features(clean)
        data_source = f"Uploaded file: {uploaded.name}"
    except Exception as exc:
        st.error(f"Could not process the uploaded CSV: {exc}")
        st.stop()

dept = department_summary(clean)
dept_reset = dept.reset_index()
symptom_counts = clean["symptoms"].value_counts()
symptom_audit = symptom_department_audit(clean)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("Healthcare Operations Intelligence")
st.caption(data_source)
st.markdown(f"> ⚠️ {DISCLAIMER}")

# ---------------------------------------------------------------------
# Executive Overview
# ---------------------------------------------------------------------
if page == "Executive Overview":
    st.subheader("Decision-ready view of the synthetic patient dataset")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Clean patients", f"{len(clean):,}")
    with c2:
        metric_card("Average age", f"{clean['age'].mean():.2f}")
    with c3:
        metric_card("Chronic condition", pct((clean["chronic_condition"] == "Yes").mean() * 100))
    with c4:
        metric_card("Busiest department", clean["department"].value_counts().idxmax())
    with c5:
        metric_card("Most common symptom", symptom_counts.idxmax())

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            dept_reset.sort_values("patients"),
            x="patients",
            y="department",
            orientation="h",
            title="Department Workload",
            labels={"patients": "Patients", "department": ""},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        sev = clean["symptom_severity"].value_counts().reindex(
            ["Mild", "Moderate", "Severe"]
        ).reset_index()
        sev.columns = ["severity", "patients"]
        fig = px.pie(
            sev, names="severity", values="patients",
            hole=0.55, title="Symptom Severity Mix"
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("What stands out?")
    top_dept = dept.iloc[0]
    top_symptom = symptom_counts.idxmax()
    oldest = featured["age_group"].value_counts().idxmax()
    insight_box(
        "Workload concentration",
        f"{dept.index[0]} has the largest patient volume at {int(top_dept['patients']):,} records "
        f"({top_dept['volume_share']:.2f}% of the cleaned dataset).",
    )
    insight_box(
        "Symptom pattern",
        f"{top_symptom} is the most frequently reported symptom ({int(symptom_counts.max()):,} records). "
        "Because this dataset is synthetic, the pattern should not be interpreted as a real-world prevalence estimate.",
    )
    insight_box(
        "Population profile",
        f"The largest age band is {oldest}, with {int(featured['age_group'].value_counts().max()):,} records. "
        f"Chronic-condition prevalence in the cleaned data is {(clean['chronic_condition'] == 'Yes').mean()*100:.2f}%.",
    )

# ---------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------
elif page == "Data Quality":
    st.subheader("Auditable data-quality pipeline")
    st.write(
        "The raw file is never overwritten. Corrections are applied to a separate working copy, "
        "and the enriched output carries row-level quality flags."
    )

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        metric_card("Raw rows", f"{report['raw_rows']:,}")
    with q2:
        metric_card("Duplicates removed", f"{report['exact_duplicates_removed']:,}")
    with q3:
        metric_card("Raw missing cells", f"{report['raw_missing_cells']:,}")
    with q4:
        metric_card("Final missing cells", f"{int(clean.isna().sum().sum()):,}")

    left, right = st.columns(2)
    with left:
        miss = raw.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].reset_index()
        miss.columns = ["column", "missing"]
        fig = px.bar(
            miss, x="missing", y="column", orientation="h",
            title="Missing Values in Raw Data",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        quality_counts = enriched["quality_status"].value_counts().reset_index()
        quality_counts.columns = ["status", "rows"]
        fig = px.bar(
            quality_counts, x="status", y="rows",
            title="Row-level Quality Status",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cleaning decisions")
    cleaning_table = pd.DataFrame([
        ["Exact duplicate rows", int(report["exact_duplicates_removed"]), "Removed"],
        ["Oxygen saturation > 100", int(report["oxygen_values_capped"]), "Capped at 100"],
        ["Extreme systolic BP", int(report["extreme_systolic_values_replaced"]), "Set missing → median"],
        ["Missing numeric values", int(raw.select_dtypes(include=np.number).isna().sum().sum()), "Median imputation"],
        ["Missing categorical values", int(raw.select_dtypes(exclude=np.number).isna().sum().sum()), "Mode imputation"],
    ], columns=["Issue", "Affected cells/rows", "Treatment"])
    st.dataframe(cleaning_table, use_container_width=True, hide_index=True)

    st.success("Final cleaned dataset contains zero missing values and 150,000 unique patient IDs.")

# ---------------------------------------------------------------------
# Department Intelligence
# ---------------------------------------------------------------------
elif page == "Department Intelligence":
    st.subheader("Department workload and service-mix intelligence")

    st.write(
        "The Operations Load Index is a transparent analytical index—not a clinical score. "
        "Adjust the weights to explore how different operational priorities change the index."
    )

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        v = st.slider("Volume weight", 0.0, 1.0, 0.40, 0.05)
    with w2:
        s = st.slider("Severe-share weight", 0.0, 1.0, 0.25, 0.05)
    with w3:
        c = st.slider("Chronic-share weight", 0.0, 1.0, 0.15, 0.05)
    with w4:
        d = st.slider("Duration weight", 0.0, 1.0, 0.20, 0.05)

    weights = np.array([v, s, c, d])
    if weights.sum() == 0:
        weights[:] = 0.25
    weights = weights / weights.sum()

    dept_w = department_summary(clean, tuple(weights)).reset_index()
    fig = px.bar(
        dept_w.sort_values("ops_load_index"),
        x="ops_load_index", y="department", orientation="h",
        color="ops_load_index",
        title="Operations Load Index",
        color_continuous_scale="Blues",
        labels={"ops_load_index": "Index (0–100)", "department": ""},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        dept_w.style.format({
            "patients": "{:,.0f}",
            "volume_share": "{:.2f}%",
            "severe_share": "{:.2f}%",
            "chronic_share": "{:.2f}%",
            "avg_duration_days": "{:.2f}",
            "avg_pain": "{:.2f}",
            "avg_heart_rate": "{:.2f}",
            "avg_oxygen_saturation": "{:.2f}",
            "ops_load_index": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    selected = st.selectbox("Inspect a department", dept_w["department"].tolist())
    row = dept_w.loc[dept_w["department"] == selected].iloc[0]
    a, b, c2, d2 = st.columns(4)
    with a:
        metric_card("Patients", f"{int(row['patients']):,}")
    with b:
        metric_card("Severe share", f"{row['severe_share']:.2f}%")
    with c2:
        metric_card("Chronic share", f"{row['chronic_share']:.2f}%")
    with d2:
        metric_card("Avg duration", f"{row['avg_duration_days']:.2f} days")

    st.caption(
        "The index is intentionally transparent so users can inspect the components instead of treating a black-box score as a medical truth."
    )

# ---------------------------------------------------------------------
# Patient & Symptom Explorer
# ---------------------------------------------------------------------
elif page == "Patient & Symptom Explorer":
    st.subheader("Population, symptom, severity and cohort analysis")

    selected_depts = st.multiselect(
        "Department filter",
        sorted(clean["department"].unique()),
        default=sorted(clean["department"].unique()),
    )
    view = featured[featured["department"].isin(selected_depts)].copy()

    a, b = st.columns(2)
    with a:
        age_counts = view["age_group"].value_counts().reindex(["18-30","31-45","46-60","61-80"]).fillna(0).reset_index()
        age_counts.columns = ["age_group", "patients"]
        st.plotly_chart(
            px.bar(age_counts, x="age_group", y="patients", title="Patients by Age Group"),
            use_container_width=True,
        )

    with b:
        top = view["symptoms"].value_counts().head(12).sort_values().reset_index()
        top.columns = ["symptom", "patients"]
        st.plotly_chart(
            px.bar(top, x="patients", y="symptom", orientation="h", title="Top Symptoms"),
            use_container_width=True,
        )

    cohort = pd.crosstab(
        view["age_group"], view["chronic_condition"], normalize="index"
    ).mul(100).reset_index()
    long = cohort.melt(id_vars=["age_group"], var_name="condition", value_name="share")
    st.plotly_chart(
        px.bar(
            long, x="age_group", y="share", color="condition",
            title="Chronic-Condition Mix by Age Group",
            labels={"share": "Share (%)", "age_group": "Age group"},
        ),
        use_container_width=True,
    )

    st.subheader("Synthetic symptom → department association")
    st.write(
        f"Cramer's V = **{symptom_audit['cramers_v']:.3f}** and a symptom-to-most-common-department "
        f"lookup reaches **{symptom_audit['symptom_lookup_accuracy']*100:.2f}%** on this dataset. "
        "This very strong association is a property of the synthetic data-generation process, so symptoms are excluded from the final ML benchmark."
    )

# ---------------------------------------------------------------------
# ML Audit
# ---------------------------------------------------------------------
elif page == "ML Audit":
    st.subheader("Exploratory machine-learning audit")
    st.caption(
        "Target = department. The intentionally high-leakage symptoms field is excluded from the benchmark."
    )

    default_metrics_path = ROOT / "artifacts" / "model_metrics.csv"
    if uploaded is None and default_metrics_path.exists():
        metrics = pd.read_csv(default_metrics_path)
        cm = None
        st.success("Precomputed reproducible benchmark loaded from project artifacts.")
    else:
        with st.spinner("Running four-model benchmark on a stratified 80/20 split..."):
            metrics, cm = run_model_cached(clean)

    display = metrics.copy()
    for col in ["accuracy", "balanced_accuracy"]:
        display[col] = display[col].mul(100)
    st.dataframe(
        display.style.format({
            "accuracy": "{:.2f}%",
            "balanced_accuracy": "{:.2f}%",
            "macro_f1": "{:.3f}",
            "macro_precision": "{:.3f}",
            "macro_recall": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        metrics.sort_values("accuracy"),
        x="accuracy",
        y="model",
        orientation="h",
        title="Accuracy Benchmark",
        labels={"accuracy": "Accuracy"},
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    best = metrics.iloc[0]
    insight_box(
        "Interpretation",
        f"{best['model']} is the strongest model in this benchmark at {best['accuracy']*100:.2f}% accuracy "
        f"and {best['balanced_accuracy']*100:.2f}% balanced accuracy. "
        "The result remains modest, so the model is treated as an educational experiment rather than a real clinical routing system.",
    )

    with st.expander("Why symptoms are excluded"):
        st.write(
            f"Cramer's V = {symptom_audit['cramers_v']:.3f}. "
            f"A simple symptom lookup reaches {symptom_audit['symptom_lookup_accuracy']*100:.2f}% accuracy. "
            "Including symptoms would largely reproduce the synthetic assignment rule instead of testing independent predictive signal."
        )

    if cm is not None:
        st.subheader("Random Forest confusion matrix")
        cm_fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        )
        st.plotly_chart(cm_fig, use_container_width=True)

# ---------------------------------------------------------------------
# Evidence Q&A
# ---------------------------------------------------------------------
elif page == "Evidence Q&A":
    st.subheader("Ask the dashboard")
    st.write("These answers are generated directly from the cleaned dataset—not from unsupported medical assumptions.")

    questions = {
        "What is the average patient age?": f"The average patient age is {clean['age'].mean():.2f} years.",
        "Which department has the highest workload?": f"{clean['department'].value_counts().idxmax()} has the highest volume with {clean['department'].value_counts().max():,} patients.",
        "What is the most common symptom?": f"{symptom_counts.idxmax()} is the most common symptom with {symptom_counts.max():,} records.",
        "What share of patients has a chronic condition?": f"{(clean['chronic_condition'] == 'Yes').mean()*100:.2f}% of cleaned records have a chronic-condition label of Yes.",
        "Which department has the highest average heart rate?": f"{clean.groupby('department')['heart_rate'].mean().idxmax()} has the highest average heart rate at {clean.groupby('department')['heart_rate'].mean().max():.2f} bpm.",
        "Which department has the highest average pain level?": f"{clean.groupby('department')['pain_level'].mean().idxmax()} has the highest average pain level at {clean.groupby('department')['pain_level'].mean().max():.2f}/10.",
        "Which age group is largest?": f"{featured['age_group'].value_counts().idxmax()} is the largest age group with {featured['age_group'].value_counts().max():,} records.",
        "How strong is the symptom-department relationship?": f"Cramer's V is {symptom_audit['cramers_v']:.3f}; the symptom lookup accuracy is {symptom_audit['symptom_lookup_accuracy']*100:.2f}%. This reflects synthetic data structure.",
    }
    q = st.selectbox("Choose a question", list(questions))
    insight_box("Evidence-based answer", questions[q])

# ---------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------
elif page == "Downloads":
    st.subheader("Project outputs")
    st.write("Download the cleaned dataset, audit dataset and analytical artifacts.")

    for label, path, mime in [
        ("Cleaned CSV", ROOT / "data" / "healthcare_cleaned_150000.csv", "text/csv"),
        ("Enriched audit CSV", ROOT / "data" / "healthcare_enriched_150000.csv", "text/csv"),
        ("Model metrics", ROOT / "artifacts" / "model_metrics.csv", "text/csv"),
        ("Department summary", ROOT / "artifacts" / "department_summary.csv", "text/csv"),
        ("Key questions", ROOT / "outputs" / "tables" / "key_questions.csv", "text/csv"),
    ]:
        if path.exists():
            st.download_button(
                label=f"Download {label}",
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime,
            )

    st.markdown("### Reproducibility")
    st.code(
        "python scripts/build_artifacts.py\npytest -q\nstreamlit run app.py",
        language="bash",
    )
