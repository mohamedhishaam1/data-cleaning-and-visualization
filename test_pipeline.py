"""
test_pipeline.py
Automated test suite verifying data cleaning, statistical analysis, and visualization generation.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for headless test
import matplotlib.pyplot as plt

from data_cleaning import run_full_cleaning_pipeline, inspect_dataset
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

def test_pipeline():
    print("=== 1. Testing Raw Data Loading & Inspection ===")
    data_path = os.path.join(os.path.dirname(__file__), "data", "sample_sales.csv")
    assert os.path.exists(data_path), f"File not found: {data_path}"
    raw_df = pd.read_csv(data_path)
    print(f"Loaded raw dataset with shape: {raw_df.shape}")
    
    inspect_res = inspect_dataset(raw_df)
    print(f"Raw Missing Cells: {inspect_res['missing_cells']}")
    print(f"Raw Duplicates: {inspect_res['duplicates']}")
    assert inspect_res["missing_cells"] > 0, "Expected raw missing cells"
    assert inspect_res["duplicates"] > 0, "Expected raw duplicates"

    print("\n=== 2. Testing Data Cleaning Pipeline ===")
    cleaned_df, summary = run_full_cleaning_pipeline(
        raw_df=raw_df,
        numeric_strategy="median",
        categorical_strategy="unknown",
        outlier_method="clip",
        iqr_factor=1.5
    )
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Duplicates removed: {summary['duplicates_removed']}")
    print(f"Invalid records removed: {summary['invalid_records_removed']}")
    print(f"Final missing cells: {summary['final_missing_cells']}")
    assert summary["final_missing_cells"] == 0, "Expected 0 missing cells after cleaning"
    assert summary["final_duplicates"] == 0, "Expected 0 duplicates after cleaning"

    print("\n=== 3. Testing Statistical Analysis ===")
    kpis = calculate_kpis(cleaned_df)
    print(f"KPIs: Total Sales=${kpis['total_sales']}, Orders={kpis['total_orders']}, AOV=${kpis['avg_order_value']}")
    assert kpis["total_sales"] > 0, "Total sales should be > 0"
    assert kpis["total_orders"] > 0, "Total orders should be > 0"

    cat_df = analyze_category_sales(cleaned_df)
    print("Category breakdown:\n", cat_df.head(3))
    assert not cat_df.empty, "Category analysis returned empty"

    reg_df = analyze_regional_sales(cleaned_df)
    print("Regional breakdown:\n", reg_df.head(3))
    assert not reg_df.empty, "Regional analysis returned empty"

    prod_df = analyze_top_products(cleaned_df)
    print("Top products:\n", prod_df.head(3))
    assert not prod_df.empty, "Top products returned empty"

    trend_df = analyze_monthly_trends(cleaned_df)
    print("Monthly trends:\n", trend_df.head(3))
    assert not trend_df.empty, "Monthly trends returned empty"

    insights = generate_business_insights(cleaned_df)
    print(f"Generated {len(insights)} business insights.")
    assert len(insights) > 0, "Expected business insights"

    print("\n=== 4. Testing Visualizations ===")
    fig1 = plot_sales_by_product(cleaned_df)
    assert isinstance(fig1, plt.Figure)
    fig2 = plot_category_distribution(cleaned_df)
    assert isinstance(fig2, plt.Figure)
    fig3 = plot_monthly_sales_trend(cleaned_df)
    assert isinstance(fig3, plt.Figure)
    fig4 = plot_numerical_distribution(cleaned_df, "Total Sales")
    assert isinstance(fig4, plt.Figure)
    fig5 = plot_outlier_boxplot(cleaned_df)
    assert isinstance(fig5, plt.Figure)
    fig6 = plot_correlation_heatmap(cleaned_df)
    assert isinstance(fig6, plt.Figure)
    fig7 = plot_regional_sales(cleaned_df)
    assert isinstance(fig7, plt.Figure)
    plt.close("all")

    print("\n=== All Tests Passed Successfully! ===")

if __name__ == "__main__":
    test_pipeline()
