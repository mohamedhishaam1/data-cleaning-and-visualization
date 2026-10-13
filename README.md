# Data Cleaning and Visualization Using Python

A complete, production-grade data analysis and visualization system built with **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **Streamlit**.

This project provides an automated data quality audit and cleaning pipeline that diagnoses and fixes missing values, duplicates, outliers (IQR), inconsistent formatting, and invalid entries, paired with an interactive executive dashboard for visual business intelligence.

---

## 🌟 Key Features

1. **Intelligent Data Cleaning Pipeline (`data_cleaning.py`)**:
   - **Data Profiling**: High-level metrics on missingness, duplicates, data types, and shape.
   - **Formatting & Type Standardization**: Cleans currency symbols (`$`, `,`), standardizes messy casing, and safely parses mixed date formats (`YYYY-MM-DD`, `MM/DD/YYYY`, `DD-MM-YYYY`).
   - **Missing Value Handling**: Configurable strategies for numerical (`median`, `mean`, `drop`) and categorical (`unknown`, `mode`, `drop`) columns.
   - **Duplicate Removal**: Identifies and eliminates redundant records.
   - **Outlier Treatment via IQR**: Computes interquartile fences ($Q1 - 1.5 \times IQR$, $Q3 + 1.5 \times IQR$) with options to clip (cap), remove, or keep anomalies.
   - **Record Validation**: Purges corrupt records with negative quantities or zero prices, and reconciles calculated total sales.
   - **Before-and-After Audit**: Side-by-side comparison of dataset health metrics before and after cleaning.

2. **Statistical Analysis & KPIs (`data_analysis.py`)**:
   - High-level business metrics: Total Revenue, Total Orders, Average Order Value (AOV), and Units Sold.
   - Category and regional sales breakdowns with percentage market share.
   - Top-performing products ranked by sales and volume.
   - Monthly revenue trajectory and Month-over-Month (MoM) growth rates.
   - Automated natural-language executive findings generator.

3. **Publication-Quality Visualizations (`visualization.py`)**:
   - Top Products Horizontal Bar Chart with tip labels.
   - Category Revenue Donut Chart.
   - Monthly Sales Trend Line Chart with shaded area.
   - Numerical Distribution Histograms with KDE and Mean/Median markers.
   - Multi-feature Outlier Detection Boxplots.
   - Correlation Heatmap for numerical variables.
   - Regional Revenue Comparison Bar Chart.

4. **Interactive Web Dashboard (`app.py`)**:
   - Seamless toggling between realistic built-in sample data and custom CSV file uploads.
   - Real-time sidebar controls for data cleaning parameters and dynamic filters.
   - Tabbed layout:
     - `📋 1. Raw Data Inspection`
     - `🧹 2. Cleaning & Before/After`
     - `📊 3. Statistical Analysis & KPIs`
     - `📈 4. Data Visualizations`
     - `💡 5. Business Insights & Export`
   - One-click downloads for Cleaned CSV datasets and Markdown summary reports.

---

## 📂 Project Structure

```text
data-cleaning-visualization/
├── app.py                      # Main Streamlit interactive dashboard application
├── data_cleaning.py            # Data auditing, cleaning, and preprocessing functions
├── data_analysis.py            # Statistical computations, aggregations, and insights
├── visualization.py            # Matplotlib and Seaborn plotting routines
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation and guide
└── data/
    ├── sample_sales.csv        # Realistic raw dataset with missing values, outliers, & duplicates
    └── generate_sample_data.py # Script used to generate raw benchmark dataset
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Install Dependencies
Navigate to the project root directory and run:

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
Start the application with:

```bash
streamlit run app.py
```

The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 🔄 Data Cleaning Workflow

```text
[Raw Dataset]
      │
      ▼
1. Clean Column Headers (strip whitespace, normalize names)
      │
      ▼
2. Currency & Type Parsing (strip '$' and commas, convert to numeric)
      │
      ▼
3. Text & Date Normalization (Title Case, parse mixed date formats)
      │
      ▼
4. Filter Invalid Records (remove Quantity <= 0 or Unit Price <= 0)
      │
      ▼
5. Missing Value Treatment (Impute median/mean/mode or fill 'Unknown')
      │
      ▼
6. Reconcile Calculated Fields (Total Sales = Quantity * Unit Price)
      │
      ▼
7. Duplicate Purge (Drop exact duplicate records)
      │
      ▼
8. IQR Outlier Handling (Clip or remove values exceeding 1.5 * IQR)
      │
      ▼
[Cleaned & Validated Dataset Ready for Analytics & Modeling]
```

---

## 📊 Major Findings from the Sample Dataset

Using the generated benchmark sales dataset (`data/sample_sales.csv`):
- **Data Quality Issues Resolved**:
  - Found and eliminated **15 duplicate records**.
  - Imputed missing values across `Customer Name`, `Region`, `Quantity`, and `Unit Price`.
  - Filtered invalid negative quantity transactions.
  - Handled extreme quantity and price outliers using IQR clipping.
- **Top Category**: **Technology** generated the highest overall revenue share, driven by premium laptops and monitors.
- **Star Product**: High-ticket products such as the **Apple MacBook Pro** and **Dell XPS 15** represented the largest share of total dollar sales.
- **Peak Sales Months**: Clear seasonal upticks in Q4 and mid-year promotional periods.
- **Regional Balance**: Sales were fairly evenly distributed across North, South, East, and West, with North and West leading in average ticket size.

---

## 🛡️ Error Handling & Robustness

- **CSV Validation**: Validates file types and prevents crashes on corrupt or empty uploaded files.
- **Graceful Fallbacks**: Missing optional columns are detected and skipped without terminating the application.
- **Zero-Division & Empty State Protection**: KPI cards and visualization functions display informative placeholders when filtered down to zero records.
