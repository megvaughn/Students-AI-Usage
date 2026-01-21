# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Optional: kagglehub support
try:
    import kagglehub
    HAS_KAGGLEHUB = True
except Exception:
    HAS_KAGGLEHUB = False

st.set_page_config(page_title="Students AI Usage Dashboard", layout="wide")

AI_COLOR = "lightseagreen"
NO_AI_COLOR = "orchid"

st.title("Students’ AI Usage and Academic Performance")
st.caption("Interactive dashboard to explore AI usage patterns and academic outcomes (synthetic dataset).")

# ---------------------------
# Helpers
# ---------------------------
def normalize_yes_no(x):
    if pd.isna(x):
        return "No"
    s = str(x).strip().lower()
    if s in {"yes", "true", "1", "y"}:
        return "Yes"
    if s in {"no", "false", "0", "n"}:
        return "No"
    return str(x).title()

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize categories
    if "uses_ai" in df.columns:
        df["uses_ai"] = df["uses_ai"].apply(normalize_yes_no)

    for col in ["education_level", "ai_tools_used", "purpose_of_ai"]:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str).str.strip()

    # numeric conversion
    for col in ["age", "study_hours_per_day", "grades_before_ai", "grades_after_ai", "daily_screen_time_hours"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # engineered features
    if {"grades_before_ai", "grades_after_ai"}.issubset(df.columns):
        df["grade_change"] = df["grades_after_ai"] - df["grades_before_ai"]
        df["improved"] = (df["grade_change"] > 0).astype(int)

    return df

# ---------------------------
# Data loading
# ---------------------------
@st.cache_data
def load_data_local(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_data_kagglehub() -> pd.DataFrame:
    if not HAS_KAGGLEHUB:
        raise RuntimeError("kagglehub is not installed. Run: pip install kagglehub")
    path = kagglehub.dataset_download("aminasalamt/students-ai-usage-and-academic-performance")
    csv_path = Path(path) / "students_ai_usage.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find students_ai_usage.csv in: {path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df

# ---------------------------
# Sidebar: data source
# ---------------------------
st.sidebar.header("Data Source")

source = st.sidebar.radio(
    "Choose dataset source",
    options=["Local CSV", "KaggleHub (download)"],
    index=1  # default to KaggleHub so it works immediately
)

df = None

if source == "Local CSV":
    csv_path = st.sidebar.text_input("Path to students_ai_usage.csv", value="data/students_ai_usage.csv")

    if not Path(csv_path).exists():
        st.sidebar.error(f"File not found: {csv_path}")
        st.sidebar.info("Fix the path or switch to KaggleHub (download).")
        st.stop()

    try:
        df = load_data_local(csv_path)
        st.sidebar.success("Loaded local CSV ✅")
    except Exception as e:
        st.sidebar.error("Could not read CSV:")
        st.sidebar.exception(e)
        st.stop()

else:
    st.sidebar.write("Using KaggleHub…")
    if not HAS_KAGGLEHUB:
        st.sidebar.error("kagglehub is not installed in this environment.")
        st.sidebar.code("pip install kagglehub")
        st.stop()

    try:
        # Download and locate CSV robustly
        path = kagglehub.dataset_download("aminasalamt/students-ai-usage-and-academic-performance")
        base = Path(path)
        st.sidebar.write(f"Downloaded to: {base}")

        # Search recursively for the CSV (handles structure differences)
        matches = list(base.rglob("students_ai_usage.csv"))
        if len(matches) == 0:
            matches = list(base.rglob("*.csv"))

        if len(matches) == 0:
            st.sidebar.error("No CSV found inside the downloaded dataset folder.")
            st.sidebar.write(f"Contents of {base}:")
            st.sidebar.write([p.name for p in base.rglob("*")][:50])
            st.stop()

        csv_path = matches[0]
        st.sidebar.write(f"Using file: {csv_path.name}")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        st.sidebar.success("Loaded KaggleHub dataset ✅")

    except Exception as e:
        st.sidebar.error("KaggleHub load failed:")
        st.sidebar.exception(e)
        st.stop()

# Clean after successful load
df = clean_df(df)

# ---------------------------
# Sidebar: filters
# ---------------------------
st.sidebar.header("Filters")

# Safer unique lists (in case of unexpected missing cols)
uses_ai_vals = ["All", "Yes", "No"] if "uses_ai" in df.columns else ["All"]
edu_vals = ["All"] + sorted(df["education_level"].dropna().unique().tolist()) if "education_level" in df.columns else ["All"]
tool_vals = ["All"] + sorted(df["ai_tools_used"].dropna().unique().tolist()) if "ai_tools_used" in df.columns else ["All"]
purpose_vals = ["All"] + sorted(df["purpose_of_ai"].dropna().unique().tolist()) if "purpose_of_ai" in df.columns else ["All"]

uses_ai = st.sidebar.selectbox("Uses AI", uses_ai_vals, index=0)
edu = st.sidebar.selectbox("Education level", edu_vals, index=0)
tool = st.sidebar.selectbox("AI tool used", tool_vals, index=0)
purpose = st.sidebar.selectbox("Purpose of AI", purpose_vals, index=0)

d = df.copy()

if uses_ai != "All" and "uses_ai" in d.columns:
    d = d[d["uses_ai"] == uses_ai]
if edu != "All" and "education_level" in d.columns:
    d = d[d["education_level"] == edu]
if tool != "All" and "ai_tools_used" in d.columns:
    d = d[d["ai_tools_used"] == tool]
if purpose != "All" and "purpose_of_ai" in d.columns:
    d = d[d["purpose_of_ai"] == purpose]

# ---------------------------
# Layout
# ---------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Preview")
    st.write(f"Rows in view: **{len(d)}**")
    st.dataframe(d.head(20), use_container_width=True)

with col2:
    st.subheader("Summary")
    if "grade_change" in d.columns:
        st.write("Grade change summary:")
        st.dataframe(d["grade_change"].describe().to_frame(name="value"), use_container_width=True)
    else:
        st.info("grade_change not available (missing grades columns).")

st.divider()

# ---------------------------
# Plot selector
# ---------------------------
st.subheader("Visualizations")

plot_type = st.selectbox(
    "Choose a plot",
    [
        "Histogram: Grades Before vs After",
        "Box + Jitter: Grade Change by AI Usage",
        "Mean ± SD: Grade Change by AI Purpose",
        "Scatter: Study Hours vs Grade Change",
        "Scatter: Screen Time vs Grade Change",
    ]
)

bins = st.slider("Histogram bins", 5, 30, 10)
alpha = st.slider("Point transparency", 0.1, 1.0, 0.6)

fig = plt.figure(figsize=(7, 4))

# Plot 1: Histogram
if plot_type == "Histogram: Grades Before vs After":
    if {"grades_before_ai", "grades_after_ai"}.issubset(d.columns):
        plt.hist(d["grades_before_ai"].dropna(), bins=bins, alpha=0.6, label="Before", color=NO_AI_COLOR)
        plt.hist(d["grades_after_ai"].dropna(), bins=bins, alpha=0.6, label="After", color=AI_COLOR)
        plt.xlabel("Grade")
        plt.ylabel("Count")
        plt.title("Grades Before vs After (Filtered)")
        plt.legend()
    else:
        st.warning("Missing grades_before_ai / grades_after_ai columns for histogram.")

# Plot 2: Box + Jitter
elif plot_type == "Box + Jitter: Grade Change by AI Usage":
    if "grade_change" not in df.columns or "uses_ai" not in df.columns:
        st.warning("Missing grade_change or uses_ai columns for boxplot.")
    else:
        # compare Yes vs No within other filters (ignore uses_ai filter here for comparison)
        d2 = df.copy()
        if edu != "All" and "education_level" in d2.columns:
            d2 = d2[d2["education_level"] == edu]
        if tool != "All" and "ai_tools_used" in d2.columns:
            d2 = d2[d2["ai_tools_used"] == tool]
        if purpose != "All" and "purpose_of_ai" in d2.columns:
            d2 = d2[d2["purpose_of_ai"] == purpose]

        data = [
            d2.loc[d2["uses_ai"] == "Yes", "grade_change"].dropna(),
            d2.loc[d2["uses_ai"] == "No", "grade_change"].dropna()
        ]

        labels = ["Uses AI", "No AI"]
        colors = [AI_COLOR, NO_AI_COLOR]

        box = plt.boxplot(data, labels=labels, patch_artist=True)
        for patch, c in zip(box["boxes"], colors):
            patch.set_facecolor(c)

        for i, (grp, c) in enumerate(zip(data, colors), start=1):
            x = np.random.normal(i, 0.05, size=len(grp))
            plt.plot(x, grp, "o", color=c, alpha=alpha)

        plt.ylabel("Grade Change")
        plt.title("Grade Change by AI Usage")

# Plot 3: Mean ± SD by Purpose
elif plot_type == "Mean ± SD: Grade Change by AI Purpose":
    if not {"uses_ai", "purpose_of_ai", "grade_change"}.issubset(d.columns):
        st.warning("Missing uses_ai, purpose_of_ai, or grade_change columns for purpose plot.")
    else:
        d_ai = d[d["uses_ai"] == "Yes"].dropna(subset=["purpose_of_ai", "grade_change"])
        if len(d_ai) == 0:
            st.warning("No AI-user rows available under current filters to plot by purpose.")
        else:
            stats_by = d_ai.groupby("purpose_of_ai")["grade_change"].agg(["mean", "std", "count"]).reset_index()
            x = stats_by["purpose_of_ai"].astype(str).tolist()
            y = stats_by["mean"].values
            yerr = stats_by["std"].values

            plt.bar(x, y, yerr=yerr, capsize=6, color=NO_AI_COLOR)
            plt.ylabel("Mean Grade Change")
            plt.title("Mean Grade Change by AI Purpose (± SD)")
            plt.xticks(rotation=25, ha="right")

            for i, n in enumerate(stats_by["count"].values):
                plt.text(i, y[i] + (yerr[i] if not np.isnan(yerr[i]) else 0) + 0.3, f"n={int(n)}",
                         ha="center", fontsize=9)

# Plot 4: Study hours scatter
elif plot_type == "Scatter: Study Hours vs Grade Change":
    if not {"study_hours_per_day", "grade_change"}.issubset(d.columns):
        st.warning("Missing study_hours_per_day or grade_change columns for scatter.")
    else:
        plt.scatter(d["study_hours_per_day"], d["grade_change"], alpha=alpha, color=NO_AI_COLOR)
        plt.xlabel("Study Hours per Day")
        plt.ylabel("Grade Change")
        plt.title("Study Hours vs Grade Change")

# Plot 5: Screen time scatter
elif plot_type == "Scatter: Screen Time vs Grade Change":
    if not {"daily_screen_time_hours", "grade_change"}.issubset(d.columns):
        st.warning("Missing daily_screen_time_hours or grade_change columns for scatter.")
    else:
        plt.scatter(d["daily_screen_time_hours"], d["grade_change"], alpha=alpha, color=NO_AI_COLOR)
        plt.xlabel("Daily Screen Time (hours)")
        plt.ylabel("Grade Change")
        plt.title("Screen Time vs Grade Change")

plt.tight_layout()
st.pyplot(fig)

st.divider()

# ---------------------------
# Quick takeaways
# ---------------------------
st.subheader("Quick Takeaways")
st.markdown(
    """
- Use the sidebar filters to explore subgroups.
- In this dataset, non-AI users typically show grade_change near 0, which can create perfect separation in classification models.
- Because the data are synthetic, patterns demonstrate methodology rather than causal effects.
"""
)

st.caption("Run this app with: `streamlit run app.py`")
