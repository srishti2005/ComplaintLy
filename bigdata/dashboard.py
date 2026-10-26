import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ComplaintLy Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background-color: #f5f7fb;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background-color: #111827;
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    /* Select box: white field with dark, readable text
       (must come AFTER the sidebar "*" rule above) */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="select"] div,
    section[data-testid="stSidebar"] div[data-baseweb="select"] input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #111827 !important;
    }

    /* Open dropdown list is rendered outside the sidebar */
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li * {
        color: #111827 !important;
    }


    /* ---------- HEADER ---------- */

    .dashboard-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 4px;
    }

    .dashboard-subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 30px;
    }


    /* ---------- KPI CARDS ---------- */

    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 22px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        min-height: 130px;
    }

    .kpi-label {
        color: #6b7280;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .kpi-value {
        color: #111827;
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    .kpi-description {
        color: #9ca3af;
        font-size: 12px;
        margin-top: 4px;
    }


    /* ---------- SECTION HEADERS ---------- */

    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #111827;
        margin-top: 30px;
        margin-bottom: 12px;
    }


    /* ---------- INFO BOX ---------- */

    .info-box {
        background: white;
        border-left: 5px solid #4f46e5;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 20px 0;
        color: #4b5563;
        font-size: 14px;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #9ca3af;
        padding: 30px 0 10px 0;
        font-size: 13px;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOCATION
# ------------------------------------------------------------
# Looks for the analytics_* Parquet folders in, in order:
#   1. $DATA_BASE (environment variable), if set
#   2. ./data/processed          (relative to where you run streamlit)
#   3. <folder of this file>/data/processed
# ============================================================

DATASET_NAMES = [
    "product",
    "company",
    "state",
    "monthly",
    "response",
    "timely",
    "issue",
]

HERE = Path(__file__).resolve().parent


def find_base():
    candidates = []
    env_base = os.getenv("DATA_BASE")
    if env_base:
        candidates.append(Path(env_base))
    candidates.append(Path("data/processed"))
    candidates.append(HERE / "data" / "processed")

    for candidate in candidates:
        if (candidate / "analytics_product").exists():
            return candidate
    return None, candidates


_found = find_base()

if not isinstance(_found, Path):
    _, searched = _found
    st.error("Analytics data not found.")
    st.write("Looked for an `analytics_product` folder in:")
    for c in searched:
        st.code(str(c))
    st.write(
        "Run the Spark jobs, copy the `analytics_*` folders there, "
        "or set the `DATA_BASE` environment variable to their location."
    )
    st.stop()

BASE_PATH = _found


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner="Loading Parquet datasets...")
def load_data(base: str):
    data = {}
    for name in DATASET_NAMES:
        data[name] = pd.read_parquet(Path(base) / f"analytics_{name}")
    return data


try:
    data = load_data(str(BASE_PATH))
except FileNotFoundError as exc:
    st.error(f"Missing dataset folder: {exc}")
    st.stop()

product_raw = data["product"]
company_raw = data["company"]
state_raw = data["state"]
monthly_df = data["monthly"].copy()
response_raw = data["response"]
timely_raw = data["timely"]
issue_raw = data["issue"]


# ============================================================
# DATA PREPARATION
# ============================================================

monthly_df["YearMonth"] = monthly_df["YearMonth"].astype(str)
monthly_df["Year"] = monthly_df["YearMonth"].str[:4].astype(int)

years = sorted(int(y) for y in monthly_df["Year"].unique())


def year_col(df):
    """Return the name of a 'year' column (any case), or None."""
    for col in df.columns:
        if col.lower() == "year":
            return col
    return None


def prep(df, key):
    """
    Apply the sidebar year filter when the dataset has a year column,
    then aggregate to one row per `key`, sorted by complaint_count.
    Returns (dataframe, year_aware).

    Datasets without a year column are returned unfiltered
    (year_aware=False) and the UI says so.
    """
    yc = year_col(df)

    if yc is None:
        return df.sort_values("complaint_count", ascending=False), False

    d = df
    if selected_year != "All Years":
        d = d[d[yc].astype(int) == int(selected_year)]

    d = d.groupby(key, as_index=False)["complaint_count"].sum()
    return d.sort_values("complaint_count", ascending=False), True


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:25px;
            font-weight:800;
            color:white;
            margin-bottom:5px;
        ">
        📊 ComplaintLy
        </div>

        <div style="
            color:#9ca3af;
            font-size:13px;
            margin-bottom:30px;
        ">
        Intelligence Dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Dashboard Controls")

    selected_year = st.selectbox(
        "Select Year",
        ["All Years"] + years,
    )

    top_n = st.slider(
        "Number of Items",
        min_value=5,
        max_value=15,
        value=10,
    )

    st.markdown("---")

    st.markdown(
        """
        **Data Pipeline**

        🗂️ CFPB Complaint Dataset

        ↓

        ⚡ PySpark ETL

        ↓

        🗃️ Partitioned Parquet

        ↓

        🔎 Spark SQL

        ↓

        📊 Streamlit
        """
    )

    st.markdown("---")

    st.caption("18.1M+ records processed using PySpark")
    st.caption(f"Data folder: {BASE_PATH}")


# ============================================================
# APPLY YEAR FILTER
# ============================================================

if selected_year == "All Years":
    filtered_monthly = monthly_df.copy()
else:
    filtered_monthly = monthly_df[monthly_df["Year"] == selected_year].copy()

product_df, product_aware = prep(product_raw, "Product")
company_df, company_aware = prep(company_raw, "Company")
state_df, state_aware = prep(state_raw, "State")
issue_df, issue_aware = prep(issue_raw, "Issue")
response_df, response_aware = prep(response_raw, "response")
timely_df, timely_aware = prep(timely_raw, "timely_response")

year_label = "All years" if selected_year == "All Years" else str(selected_year)


def scope_note(aware):
    """Tell the user when the year filter does not apply to a view."""
    if selected_year != "All Years" and not aware:
        st.caption(
            "ℹ️ Showing all years: this dataset has no Year column yet. "
            "Add Year to the Spark aggregation to enable the filter."
        )


def hbar(df, label_col, n, height=650, y_title=""):
    """Horizontal bar chart of the top-n rows by complaint_count."""
    if df.empty:
        st.info("No data for the selected year.")
        return

    chart = df.head(n).sort_values("complaint_count")

    fig = px.bar(
        chart,
        x="complaint_count",
        y=label_col,
        orientation="h",
        text="complaint_count",
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
    )

    fig.update_layout(
        height=height,
        template="plotly_white",
        xaxis_title="Complaint Volume",
        yaxis_title=y_title,
        margin=dict(l=20, r=100, t=30, b=20),
    )

    st.plotly_chart(fig, width="stretch")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">ComplaintLy Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    "Large-Scale Consumer Complaint Analytics Platform"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    <b>Big Data Pipeline:</b>
    Consumer complaint records are processed using PySpark,
    transformed into partitioned Parquet datasets, analyzed using
    Spark SQL, and visualized through this dashboard.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# KPI SECTION
# ------------------------------------------------------------
# Total complaints:
#   - product data has a Year column  -> filtered product total
#   - All Years                        -> product total
#   - one year, no Year in product     -> monthly total for that year
# ============================================================

if product_aware or selected_year == "All Years":
    total_complaints = int(product_df["complaint_count"].sum())
else:
    total_complaints = int(filtered_monthly["complaint_count"].sum())

total_products = len(product_df)
total_companies = len(company_df)
total_states = len(state_df)

st.markdown(
    '<div class="section-title">📌 Dataset Overview</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Complaints</div>
            <div class="kpi-value">{total_complaints:,}</div>
            <div class="kpi-description">Records processed ({year_label})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Product Categories</div>
            <div class="kpi-value">{total_products}</div>
            <div class="kpi-description">Complaint categories</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Top Companies</div>
            <div class="kpi-value">{total_companies}</div>
            <div class="kpi-description">Companies in analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Top States</div>
            <div class="kpi-value">{total_states}</div>
            <div class="kpi-description">States in analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Overview",
        "📊 Products",
        "🏢 Companies",
        "🗺️ Geography",
        "⏱️ Responses",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">📈 Complaint Trend</div>',
        unsafe_allow_html=True,
    )

    filtered_monthly = filtered_monthly.sort_values("YearMonth")

    if filtered_monthly.empty:
        st.info("No monthly data for the selected year.")
    else:
        fig = px.line(
            filtered_monthly,
            x="YearMonth",
            y="complaint_count",
            markers=True,
        )

        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=5),
        )

        fig.update_layout(
            height=480,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Month",
            yaxis_title="Complaint Volume",
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig, width="stretch")

    st.markdown(
        '<div class="section-title">🔎 Quick Insights</div>',
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        if product_df.empty:
            st.info("No product data for the selected year.")
        else:
            top_product = product_df.iloc[0]
            st.info(
                f"**Highest complaint product**\n\n"
                f"{top_product['Product']}\n\n"
                f"**{int(top_product['complaint_count']):,} complaints**"
            )

    with q2:
        if company_df.empty:
            st.info("No company data for the selected year.")
        else:
            top_company = company_df.iloc[0]
            st.info(
                f"**Highest complaint-volume company in the analyzed list**\n\n"
                f"{top_company['Company']}\n\n"
                f"**{int(top_company['complaint_count']):,} complaints**"
            )

    with q3:
        if state_df.empty:
            st.info("No state data for the selected year.")
        else:
            top_state = state_df.iloc[0]
            st.info(
                f"**Highest complaint-volume state in the analyzed list**\n\n"
                f"{top_state['State']}\n\n"
                f"**{int(top_state['complaint_count']):,} complaints**"
            )


# ============================================================
# TAB 2 — PRODUCTS
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">📊 Product Analysis</div>',
        unsafe_allow_html=True,
    )

    scope_note(product_aware)
    hbar(product_df, "Product", top_n)

    st.markdown("### Product Data")

    st.dataframe(
        product_df.head(top_n),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# TAB 3 — COMPANIES
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">🏢 Company Analysis</div>',
        unsafe_allow_html=True,
    )

    scope_note(company_aware)
    hbar(company_df, "Company", top_n)

    st.warning(
        "Complaint volume does not directly indicate company performance. "
        "Counts can be affected by company size, customer base and market share."
    )


# ============================================================
# TAB 4 — GEOGRAPHY
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">🗺️ Geographic Analysis</div>',
        unsafe_allow_html=True,
    )

    scope_note(state_aware)
    hbar(state_df, "State", top_n, height=600, y_title="State")


# ============================================================
# TAB 5 — RESPONSES
# ============================================================

with tab5:

    st.markdown(
        '<div class="section-title">⏱️ Response Analysis</div>',
        unsafe_allow_html=True,
    )

    scope_note(timely_aware and response_aware)

    col1, col2 = st.columns(2)

    with col1:
        if timely_df.empty:
            st.info("No data for the selected year.")
        else:
            fig = px.pie(
                timely_df,
                names="timely_response",
                values="complaint_count",
                hole=0.55,
            )

            fig.update_layout(
                height=500,
                template="plotly_white",
                title="Timely Response Distribution",
            )

            st.plotly_chart(fig, width="stretch")

    with col2:
        if response_df.empty:
            st.info("No data for the selected year.")
        else:
            response_chart = response_df.sort_values(
                "complaint_count",
                ascending=True,
            )

            fig = px.bar(
                response_chart,
                x="complaint_count",
                y="response",
                orientation="h",
                text="complaint_count",
            )

            fig.update_traces(
                texttemplate="%{text:,.0f}",
                textposition="outside",
            )

            fig.update_layout(
                height=500,
                template="plotly_white",
                title="Company Response Distribution",
                xaxis_title="Complaints",
                yaxis_title="Response",
            )

            st.plotly_chart(fig, width="stretch")

    st.markdown(
        '<div class="section-title">🔎 Top Complaint Issues</div>',
        unsafe_allow_html=True,
    )

    scope_note(issue_aware)
    hbar(issue_df, "Issue", top_n)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ComplaintLy Intelligence • PySpark • Spark SQL • Parquet • Streamlit
        <br>
        Big Data Analytics Mini Project
    </div>
    """,
    unsafe_allow_html=True,
)