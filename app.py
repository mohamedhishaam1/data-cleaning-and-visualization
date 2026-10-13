"""
app.py
Interactive Streamlit Dashboard for Data Cleaning & Visualization.
Features:
- Seamless CSV upload or 1-click loading of realistic sample dataset
- Configurable Data Cleaning Pipeline (Missing values, Duplicates, Outliers via IQR)
- Before vs. After Cleaning Inspection and Comparative Metrics
- Dynamic Filtering (Category, Region, Date range)
- Statistical KPIs and Aggregated Tables
- Modern Matplotlib & Seaborn Visualizations
- Automated Executive Business Insights
- Export options for Cleaned CSV and Analysis Report
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime

# Import custom modules
from data_cleaning import (
    inspect_dataset,
    run_full_cleaning_pipeline,
    detect_outliers_iqr
)
from data_analysis import (
    calculate_kpis,
    analyze_top_products,
    analyze_category_sales,
    analyze_regional_sales,
    analyze_monthly_trends,
    generate_business_insights
)
from visualization import (
    plot_sales_by_product,
    plot_category_distribution,
    plot_monthly_sales_trend,
    plot_numerical_distribution,
    plot_outlier_boxplot,
    plot_correlation_heatmap,
    plot_regional_sales
)

# ---------------------------------------------------------
# Page Configuration & UI Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Data Cleaning & Visualization Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, professional typography & cards
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .clean-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #DBEAFE;
        color: #1E40AF;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar: Dataset Selection & Cleaning Parameters
# ---------------------------------------------------------
st.sidebar.title("🛠️ Control Center")
st.sidebar.markdown("Configure data sources, cleaning rules, and dynamic filters.")

# Data Source Selection
st.sidebar.subheader("1. Data Source")
sample_data_path = os.path.join(os.path.dirname(__file__), "data", "sample_sales.csv")

data_source = st.sidebar.radio(
    "Choose Data Source:",
    options=["Use Sample Sales Dataset", "Upload Custom CSV File"],
    index=0
)

raw_df = None

if data_source == "Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Uploaded: {uploaded_file.name} ({len(raw_df)} records)")
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")
    else:
        st.sidebar.info("Awaiting CSV file upload...")
else:
    if os.path.exists(sample_data_path):
        try:
            raw_df = pd.read_csv(sample_data_path)
            st.sidebar.success(f"Loaded: Sample Sales Dataset ({len(raw_df)} records)")
        except Exception as e:
            st.sidebar.error(f"Error loading sample dataset: {e}")
    else:
        st.sidebar.error("Sample dataset file not found at data/sample_sales.csv.")

if raw_df is None or raw_df.empty:
    st.info("👋 Welcome to **Data Cleaning & Visualization Studio**! Please select or upload a dataset in the sidebar to begin.")
    st.stop()


# ---------------------------------------------------------
# Sidebar: Data Cleaning Configuration
# ---------------------------------------------------------
st.sidebar.subheader("2. Data Cleaning Rules")

with st.sidebar.expander("⚙️ Cleaning Options", expanded=True):
    numeric_strategy = st.selectbox(
        "Numeric Missing Values:",
        options=["median", "mean", "drop"],
        index=0,
        help="Impute with median/mean or drop rows with missing numbers."
    )
    
    categorical_strategy = st.selectbox(
        "Categorical Missing Values:",
        options=["unknown", "mode", "drop"],
        index=0,
        help="Fill with 'Unknown', impute with most frequent (mode), or drop."
    )

    outlier_method = st.selectbox(
        "IQR Outlier Handling:",
        options=["clip", "remove", "keep"],
        index=0,
        help="Clip/Cap values to IQR fences, remove rows, or retain intact."
    )

    iqr_factor = st.slider("IQR Multiplier (Fence Sensitivity):", min_value=1.0, max_value=3.0, value=1.5, step=0.25)

# Execute Data Cleaning
try:
    cleaned_df, cleaning_summary = run_full_cleaning_pipeline(
        raw_df=raw_df,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
        outlier_method=outlier_method,
        iqr_factor=iqr_factor
    )
except Exception as e:
    st.error(f"Error during data cleaning pipeline: {e}")
    st.stop()


# ---------------------------------------------------------
# Sidebar: Dynamic Dashboard Filters (Based on Cleaned Data)
# ---------------------------------------------------------
st.sidebar.subheader("3. Data Filters")

filtered_df = cleaned_df.copy()

# Filter by Category
if "Category" in filtered_df.columns:
    all_categories = sorted([str(c) for c in filtered_df["Category"].dropna().unique()])
    selected_categories = st.sidebar.multiselect("Filter Category:", options=all_categories, default=all_categories)
    if selected_categories:
        filtered_df = filtered_df[filtered_df["Category"].astype(str).isin(selected_categories)]

# Filter by Region
if "Region" in filtered_df.columns:
    all_regions = sorted([str(r) for r in filtered_df["Region"].dropna().unique()])
    selected_regions = st.sidebar.multiselect("Filter Region:", options=all_regions, default=all_regions)
    if selected_regions:
        filtered_df = filtered_df[filtered_df["Region"].astype(str).isin(selected_regions)]

# Filter by Date Range
if "Order Date" in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df["Order Date"]):
    min_date = filtered_df["Order Date"].min().date()
    max_date = filtered_df["Order Date"].max().date()
    if min_date < max_date:
        selected_date_range = st.sidebar.date_input(
            "Filter Date Range:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        if isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
            start_d, end_d = selected_date_range
            filtered_df = filtered_df[
                (filtered_df["Order Date"].dt.date >= start_d) &
                (filtered_df["Order Date"].dt.date <= end_d)
            ]


# ---------------------------------------------------------
# Main Page: Header & Overview
# ---------------------------------------------------------
st.markdown('<div class="main-header">📊 Data Cleaning & Visualization Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated data auditing, intelligent preprocessing, and executive business analytics.</div>', unsafe_allow_html=True)

# Overview Quick Pills
st.markdown(f"""
<div>
    <span class="clean-pill">Raw Rows: {len(raw_df):,}</span>
    <span class="clean-pill">Cleaned Rows: {len(cleaned_df):,}</span>
    <span class="clean-pill">Duplicates Removed: {cleaning_summary['duplicates_removed']}</span>
    <span class="clean-pill">Active Filtered: {len(filtered_df):,}</span>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# Tabbed Layout
# ---------------------------------------------------------
tab_inspect, tab_clean, tab_kpis, tab_charts, tab_export = st.tabs([
    "📋 1. Raw Data Inspection",
    "🧹 2. Cleaning & Before/After",
    "📊 3. Statistical Analysis & KPIs",
    "📈 4. Data Visualizations",
    "💡 5. Business Insights & Export"
])


# =========================================================
# TAB 1: RAW DATA INSPECTION
# =========================================================
with tab_inspect:
    st.subheader("Raw Dataset Health Inspection")
    raw_info = inspect_dataset(raw_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", f"{raw_info['rows']:,}")
    with c2:
        st.metric("Total Columns", f"{raw_info['cols']}")
    with c3:
        st.metric("Missing Cells", f"{raw_info['missing_cells']:,}", f"{raw_info['missing_pct']}% of data", delta_color="inverse")
    with c4:
        st.metric("Duplicate Rows", f"{raw_info['duplicates']}", delta_color="inverse")

    st.markdown("#### 🔍 Raw Dataset Sample (First 10 Rows)")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.markdown("#### 📑 Column Health & Data Types Summary")
    st.dataframe(raw_info["col_stats_df"], use_container_width=True)


# =========================================================
# TAB 2: DATA CLEANING & COMPARISON
# =========================================================
with tab_clean:
    st.subheader("Data Cleaning Audit & Before/After Comparison")

    m1, m2, m3, m4 = st.columns(4)
    row_delta = cleaning_summary["final_rows"] - cleaning_summary["initial_rows"]
    missing_delta = cleaning_summary["final_missing_cells"] - cleaning_summary["initial_missing_cells"]

    with m1:
        st.metric("Total Rows", f"{cleaning_summary['final_rows']:,}", f"{row_delta} rows")
    with m2:
        st.metric("Missing Values", f"{cleaning_summary['final_missing_cells']}", f"{missing_delta} resolved", delta_color="inverse")
    with m3:
        st.metric("Duplicates Purged", f"{cleaning_summary['duplicates_removed']}", delta_color="inverse")
    with m4:
        st.metric("Invalid Records Purged", f"{cleaning_summary['invalid_records_removed']}", delta_color="inverse")

    st.markdown("---")
    st.markdown("#### ⚡ Actions Taken During Cleaning Pipeline")
    col_act1, col_act2 = st.columns(2)

    with col_act1:
        st.markdown("**1. Missing Value Imputation / Handling:**")
        if cleaning_summary["missing_actions"]:
            for col_name, act in cleaning_summary["missing_actions"].items():
                st.write(f"- **{col_name}**: {act}")
        else:
            st.write("No missing values required treatment.")

        st.markdown("**2. Duplicate & Validity Filtering:**")
        st.write(f"- Removed **{cleaning_summary['duplicates_removed']}** duplicate records.")
        st.write(f"- Removed **{cleaning_summary['invalid_records_removed']}** invalid records with non-positive quantities/prices.")

    with col_act2:
        st.markdown(f"**3. Outlier Treatment (Method: `{outlier_method.upper()}`):**")
        for col_name, rep in cleaning_summary["outlier_report"].items():
            st.write(f"- **{col_name}**: {rep['action']} (IQR: {rep['iqr']})")

    st.markdown("---")
    st.markdown("#### ✨ Cleaned Dataset Preview (First 10 Rows)")
    st.dataframe(cleaned_df.head(10), use_container_width=True)


# =========================================================
# TAB 3: STATISTICAL ANALYSIS & KPIS
# =========================================================
with tab_kpis:
    st.subheader("Business Metrics & Statistical Analysis")
    kpis = calculate_kpis(filtered_df)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Revenue", f"${kpis['total_sales']:,.2f}")
    with k2:
        st.metric("Total Orders", f"{kpis['total_orders']:,}")
    with k3:
        st.metric("Avg Order Value (AOV)", f"${kpis['avg_order_value']:,.2f}")
    with k4:
        st.metric("Total Quantity Sold", f"{kpis['total_quantity']:,}")

    st.write("")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### 📦 Sales Performance by Category")
        cat_df = analyze_category_sales(filtered_df)
        if not cat_df.empty:
            st.dataframe(cat_df, use_container_width=True)
        else:
            st.info("No category data available.")

    with col_t2:
        st.markdown("#### 🌍 Sales Performance by Region")
        reg_df = analyze_regional_sales(filtered_df)
        if not reg_df.empty:
            st.dataframe(reg_df, use_container_width=True)
        else:
            st.info("No regional data available.")

    st.markdown("---")
    col_t3, col_t4 = st.columns(2)

    with col_t3:
        st.markdown("#### 🏆 Top 10 Best-Selling Products")
        top_prods = analyze_top_products(filtered_df, top_n=10)
        if not top_prods.empty:
            st.dataframe(top_prods, use_container_width=True)
        else:
            st.info("No product data available.")

    with col_t4:
        st.markdown("#### 📅 Monthly Sales & Growth Trends")
        trends_df = analyze_monthly_trends(filtered_df)
        if not trends_df.empty:
            st.dataframe(trends_df[["Month_Str", "Total_Sales", "Order_Count", "MoM_Growth_%"]], use_container_width=True)
        else:
            st.info("No monthly trend data available.")


# =========================================================
# TAB 4: DATA VISUALIZATIONS
# =========================================================
with tab_charts:
    st.subheader("Visual Analytics Suite")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(plot_sales_by_product(filtered_df, top_n=8))
    with chart_col2:
        st.pyplot(plot_category_distribution(filtered_df))

    st.write("")
    st.pyplot(plot_monthly_sales_trend(filtered_df))

    st.write("")
    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.pyplot(plot_regional_sales(filtered_df))
    with chart_col4:
        st.pyplot(plot_correlation_heatmap(filtered_df))

    st.write("")
    st.markdown("#### 🔬 Numerical Distributions & Outlier Checks")
    chart_col5, chart_col6 = st.columns(2)
    with chart_col5:
        num_cols = [c for c in ["Total Sales", "Unit Price", "Quantity"] if c in filtered_df.columns]
        selected_num = st.selectbox("Select numerical field for distribution:", options=num_cols)
        if selected_num:
            st.pyplot(plot_numerical_distribution(filtered_df, selected_num))
    with chart_col6:
        st.pyplot(plot_outlier_boxplot(filtered_df))


# =========================================================
# TAB 5: BUSINESS INSIGHTS & EXPORT
# =========================================================
with tab_export:
    st.subheader("Executive Findings & Report Export")

    st.markdown("#### 🧠 Automated Business Insights")
    insights = generate_business_insights(filtered_df)
    for ins in insights:
        st.markdown(f"- {ins}")

    st.markdown("---")
    st.markdown("#### 💾 Export Artifacts")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("**Download Cleaned CSV Dataset**")
        csv_buffer = io.StringIO()
        cleaned_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_buffer.getvalue(),
            file_name=f"cleaned_sales_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with exp_col2:
        st.markdown("**Download Comprehensive Analysis Report (Markdown)**")
        kpi_dict = calculate_kpis(filtered_df)
        report_md = f"""# Data Cleaning & Sales Analytics Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Sales: ${kpi_dict['total_sales']:,.2f}
- Total Orders: {kpi_dict['total_orders']:,}
- Average Order Value (AOV): ${kpi_dict['avg_order_value']:,.2f}
- Total Units Sold: {kpi_dict['total_quantity']:,}

## Data Cleaning Summary
- Raw Rows: {cleaning_summary['initial_rows']}
- Cleaned Rows: {cleaning_summary['final_rows']}
- Duplicates Purged: {cleaning_summary['duplicates_removed']}
- Invalid Records Purged: {cleaning_summary['invalid_records_removed']}
- Missing Values Resolved: {cleaning_summary['initial_missing_cells'] - cleaning_summary['final_missing_cells']}

## Key Insights
{chr(10).join(['- ' + ins.replace('**', '') for ins in insights])}
"""
        st.download_button(
            label="📄 Download Analysis Report (.md)",
            data=report_md,
            file_name=f"sales_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>Data Cleaning and Visualization Studio • Built with Python & Streamlit</div>", unsafe_allow_html=True)
