import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ComplaintLy Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
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
""", unsafe_allow_html=True)


# ============================================================
# DATA PATHS
# ============================================================

BASE_PATH = "data/processed"

PRODUCT_PATH = f"{BASE_PATH}/analytics_product"
COMPANY_PATH = f"{BASE_PATH}/analytics_company"
STATE_PATH = f"{BASE_PATH}/analytics_state"
MONTHLY_PATH = f"{BASE_PATH}/analytics_monthly"
RESPONSE_PATH = f"{BASE_PATH}/analytics_response"
TIMELY_PATH = f"{BASE_PATH}/analytics_timely"
ISSUE_PATH = f"{BASE_PATH}/analytics_issue"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    product = pd.read_parquet(PRODUCT_PATH)
    company = pd.read_parquet(COMPANY_PATH)
    state = pd.read_parquet(STATE_PATH)
    monthly = pd.read_parquet(MONTHLY_PATH)
    response = pd.read_parquet(RESPONSE_PATH)
    timely = pd.read_parquet(TIMELY_PATH)
    issue = pd.read_parquet(ISSUE_PATH)

    return (
        product,
        company,
        state,
        monthly,
        response,
        timely,
        issue
    )


(
    product_df,
    company_df,
    state_df,
    monthly_df,
    response_df,
    timely_df,
    issue_df
) = load_data()


# ============================================================
# DATA PREPARATION
# ============================================================

monthly_df["Year"] = (
    monthly_df["YearMonth"]
    .astype(str)
    .str[:4]
    .astype(int)
)

years = sorted(
    monthly_df["Year"].unique()
)

product_df = product_df.sort_values(
    "complaint_count",
    ascending=False
)

company_df = company_df.sort_values(
    "complaint_count",
    ascending=False
)

state_df = state_df.sort_values(
    "complaint_count",
    ascending=False
)

issue_df = issue_df.sort_values(
    "complaint_count",
    ascending=False
)


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
        unsafe_allow_html=True
    )

    st.markdown("### Dashboard Controls")

    selected_year = st.selectbox(
        "Select Year",
        ["All Years"] + years
    )

    top_n = st.slider(
        "Number of Items",
        min_value=5,
        max_value=15,
        value=10
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

    st.caption(
        "18.1M+ records processed using PySpark"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">ComplaintLy Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Large-Scale Consumer Complaint Analytics Platform'
    '</div>',
    unsafe_allow_html=True
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
    unsafe_allow_html=True
)


# ============================================================
# FILTER MONTHLY DATA
# ============================================================

if selected_year == "All Years":

    filtered_monthly = monthly_df.copy()

else:

    filtered_monthly = monthly_df[
        monthly_df["Year"] == selected_year
    ].copy()


# ============================================================
# KPI SECTION
# ============================================================

total_complaints = int(
    product_df["complaint_count"].sum()
)

total_products = len(product_df)

total_companies = len(company_df)

total_states = len(state_df)


st.markdown(
    '<div class="section-title">📌 Dataset Overview</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Complaints</div>
            <div class="kpi-value">{total_complaints:,}</div>
            <div class="kpi-description">Records processed</div>
        </div>
        """,
        unsafe_allow_html=True
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
        unsafe_allow_html=True
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
        unsafe_allow_html=True
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
        unsafe_allow_html=True
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
        "⏱️ Responses"
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">📈 Complaint Trend</div>',
        unsafe_allow_html=True
    )

    filtered_monthly = filtered_monthly.sort_values(
        "YearMonth"
    )

    fig = px.line(
        filtered_monthly,
        x="YearMonth",
        y="complaint_count",
        markers=True
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=5)
    )

    fig.update_layout(
        height=480,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="Complaint Volume",
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.markdown(
        '<div class="section-title">🔎 Quick Insights</div>',
        unsafe_allow_html=True
    )

    q1, q2, q3 = st.columns(3)

    top_product = product_df.iloc[0]

    top_company = company_df.iloc[0]

    top_state = state_df.iloc[0]


    with q1:

        st.info(
            f"**Highest complaint product**\n\n"
            f"{top_product['Product']}\n\n"
            f"**{int(top_product['complaint_count']):,} complaints**"
        )


    with q2:

        st.info(
            f"**Highest complaint-volume company in the analyzed list**\n\n"
            f"{top_company['Company']}\n\n"
            f"**{int(top_company['complaint_count']):,} complaints**"
        )


    with q3:

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
        unsafe_allow_html=True
    )

    product_chart = (
        product_df
        .head(top_n)
        .sort_values("complaint_count")
    )

    fig = px.bar(
        product_chart,
        x="complaint_count",
        y="Product",
        orientation="h",
        text="complaint_count"
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        xaxis_title="Complaint Volume",
        yaxis_title="",
        margin=dict(l=20, r=100, t=30, b=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.markdown("### Product Data")

    st.dataframe(
        product_df.head(top_n),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 3 — COMPANIES
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">🏢 Company Analysis</div>',
        unsafe_allow_html=True
    )

    company_chart = (
        company_df
        .head(top_n)
        .sort_values("complaint_count")
    )

    fig = px.bar(
        company_chart,
        x="complaint_count",
        y="Company",
        orientation="h",
        text="complaint_count"
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        xaxis_title="Complaint Volume",
        yaxis_title="",
        margin=dict(l=20, r=100, t=30, b=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

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
        unsafe_allow_html=True
    )

    state_chart = (
        state_df
        .head(top_n)
        .sort_values("complaint_count")
    )

    fig = px.bar(
        state_chart,
        x="complaint_count",
        y="State",
        orientation="h",
        text="complaint_count"
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        xaxis_title="Complaint Volume",
        yaxis_title="State",
        margin=dict(l=20, r=100, t=30, b=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TAB 5 — RESPONSES
# ============================================================

with tab5:

    st.markdown(
        '<div class="section-title">⏱️ Response Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    with col1:

        fig = px.pie(
            timely_df,
            names="timely_response",
            values="complaint_count",
            hole=0.55
        )

        fig.update_layout(
            height=500,
            template="plotly_white",
            title="Timely Response Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        response_chart = (
            response_df
            .sort_values(
                "complaint_count",
                ascending=True
            )
        )

        fig = px.bar(
            response_chart,
            x="complaint_count",
            y="response",
            orientation="h",
            text="complaint_count"
        )

        fig.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside"
        )

        fig.update_layout(
            height=500,
            template="plotly_white",
            title="Company Response Distribution",
            xaxis_title="Complaints",
            yaxis_title="Response"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.markdown(
        '<div class="section-title">🔎 Top Complaint Issues</div>',
        unsafe_allow_html=True
    )

    issue_chart = (
        issue_df
        .head(top_n)
        .sort_values("complaint_count")
    )

    fig = px.bar(
        issue_chart,
        x="complaint_count",
        y="Issue",
        orientation="h",
        text="complaint_count"
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        xaxis_title="Complaint Volume",
        yaxis_title="",
        margin=dict(l=20, r=100, t=30, b=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


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
    unsafe_allow_html=True
)