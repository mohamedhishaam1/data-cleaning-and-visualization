"""
visualization.py
Matplotlib and Seaborn visualization suite.
Produces clean, presentation-ready charts:
- Bar chart for sales by product
- Donut/Pie chart for category revenue distribution
- Line chart with fill-under for monthly sales trends
- Histogram with KDE for numerical distributions
- Box plots for outlier visualization
- Correlation heatmap for numerical features
- Regional sales comparison bar chart
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

BRAND_PALETTE = ["#2563EB", "#38BDF8", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6"]


def plot_sales_by_product(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Horizontal bar chart showing the top N products by total sales.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=100)
    
    if "Product Name" not in df.columns or "Total Sales" not in df.columns or df.empty:
        ax.text(0.5, 0.5, "Insufficient data for Product Sales chart", ha="center", va="center")
        return fig

    top_df = (
        df.groupby("Product Name")["Total Sales"]
        .sum()
        .sort_values(ascending=True)
        .tail(top_n)
        .reset_index()
    )

    bars = ax.barh(top_df["Product Name"], top_df["Total Sales"], color="#2563EB", alpha=0.85, height=0.6)
    
    # Value annotations on bar tips
    max_val = top_df["Total Sales"].max()
    for bar in bars:
        w = bar.get_width()
        ax.text(w + (max_val * 0.01), bar.get_y() + bar.get_height() / 2,
                f" ${w:,.0f}", ha="left", va="center", fontsize=9, fontweight="bold", color="#1E293B")

    ax.set_title(f"Top {top_n} Best-Selling Products by Revenue", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel("Total Sales ($)", fontsize=10, labelpad=8, color="#334155")
    ax.set_ylabel("")
    ax.set_xlim(0, max_val * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_category_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Donut / Pie chart illustrating sales distribution across product categories.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)

    if "Category" not in df.columns or "Total Sales" not in df.columns or df.empty:
        ax.text(0.5, 0.5, "Insufficient data for Category Distribution chart", ha="center", va="center")
        return fig

    cat_sales = df.groupby("Category")["Total Sales"].sum().sort_values(ascending=False)
    labels = cat_sales.index
    values = cat_sales.values
    colors = BRAND_PALETTE[:len(labels)]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )

    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")
    
    for t in texts:
        t.set_fontsize(10)
        t.set_color("#1E293B")

    ax.set_title("Sales Distribution by Category", fontsize=13, fontweight="bold", pad=14, color="#0F172A")
    plt.tight_layout()
    return fig


def plot_monthly_sales_trend(df: pd.DataFrame) -> plt.Figure:
    """
    Line chart displaying monthly sales trends with data points and shaded area.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    if "Order Date" not in df.columns or "Total Sales" not in df.columns or df.empty:
        ax.text(0.5, 0.5, "Insufficient data for Monthly Trend chart", ha="center", va="center")
        return fig

    df_time = df.dropna(subset=["Order Date"]).copy()
    df_time["Order Date"] = pd.to_datetime(df_time["Order Date"], errors="coerce")
    df_time = df_time.dropna(subset=["Order Date"])

    if df_time.empty:
        ax.text(0.5, 0.5, "No valid dates found for trend chart", ha="center", va="center")
        return fig

    monthly = df_time.groupby(df_time["Order Date"].dt.to_period("M"))["Total Sales"].sum().reset_index()
    monthly["Period_Str"] = monthly["Order Date"].astype(str)

    x_vals = range(len(monthly))
    y_vals = monthly["Total Sales"]

    ax.plot(x_vals, y_vals, marker="o", color="#2563EB", linewidth=2.5, markersize=6, label="Monthly Revenue")
    ax.fill_between(x_vals, y_vals, alpha=0.15, color="#38BDF8")

    # Annotate points
    for i, txt in enumerate(y_vals):
        ax.annotate(
            f"${txt/1000:.1f}k",
            (x_vals[i], y_vals[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#1E293B"
        )

    ax.set_xticks(list(x_vals))
    ax.set_xticklabels(monthly["Period_Str"], rotation=45, ha="right", fontsize=9)
    ax.set_title("Monthly Revenue Trajectory", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
    ax.set_ylabel("Total Sales ($)", fontsize=10, labelpad=8, color="#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_numerical_distribution(df: pd.DataFrame, column: str) -> plt.Figure:
    """
    Histogram with KDE curve and mean/median reference indicators.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)

    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]) or df[column].dropna().empty:
        ax.text(0.5, 0.5, f"Column '{column}' is not numeric or has no data", ha="center", va="center")
        return fig

    data = df[column].dropna()
    mean_val = float(data.mean())
    median_val = float(data.median())

    sns.histplot(data, kde=True, ax=ax, color="#2563EB", edgecolor="white", bins=25, alpha=0.6)

    ax.axvline(mean_val, color="#EF4444", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:,.1f}")
    ax.axvline(median_val, color="#10B981", linestyle=":", linewidth=2, label=f"Median: {median_val:,.1f}")

    ax.set_title(f"Distribution of {column}", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel(column, fontsize=10, labelpad=8, color="#334155")
    ax.set_ylabel("Frequency", fontsize=10, labelpad=8, color="#334155")
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_outlier_boxplot(df: pd.DataFrame, columns: Optional[List[str]] = None) -> plt.Figure:
    """
    Boxplot showing outliers across selected numerical features.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)

    if columns is None:
        columns = [c for c in ["Quantity", "Unit Price", "Total Sales"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if not columns or df.empty:
        ax.text(0.5, 0.5, "No numeric columns available for Boxplot", ha="center", va="center")
        return fig

    # If columns have vastly different scales, plot normalized or individual
    if len(columns) == 1:
        sns.boxplot(y=df[columns[0]], ax=ax, color="#93C5FD", flierprops=dict(marker="o", color="#EF4444", markersize=6))
        ax.set_title(f"Outlier Detection Boxplot: {columns[0]}", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
        ax.set_ylabel(columns[0], fontsize=10)
    else:
        # Standardize for visual comparison if multiple columns
        sns.boxplot(data=df[columns], ax=ax, palette="Blues", flierprops=dict(marker="o", color="#EF4444", markersize=5))
        ax.set_title("Outlier Detection across Key Numerical Features", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
        ax.set_ylabel("Values", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Correlation matrix heatmap for numerical attributes.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=100)
    
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        ax.text(0.5, 0.5, "Need at least 2 numerical columns for correlation", ha="center", va="center")
        return fig

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
        linewidths=1,
        linecolor="white",
        ax=ax
    )

    ax.set_title("Correlation Heatmap (Numerical Columns)", fontsize=12, fontweight="bold", pad=12, color="#0F172A")
    plt.tight_layout()
    return fig


def plot_regional_sales(df: pd.DataFrame) -> plt.Figure:
    """
    Bar chart comparing sales and order counts across regions.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)

    if "Region" not in df.columns or "Total Sales" not in df.columns or df.empty:
        ax.text(0.5, 0.5, "Insufficient data for Regional Sales chart", ha="center", va="center")
        return fig

    reg_df = df.groupby("Region")["Total Sales"].sum().sort_values(ascending=False).reset_index()
    colors = ["#2563EB", "#38BDF8", "#10B981", "#F59E0B"][:len(reg_df)]
    
    bars = ax.bar(reg_df["Region"], reg_df["Total Sales"], color=colors, width=0.55, alpha=0.9)

    max_val = reg_df["Total Sales"].max()
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (max_val * 0.015),
                f"${h:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1E293B")

    ax.set_title("Regional Revenue Comparison", fontsize=13, fontweight="bold", pad=12, color="#0F172A")
    ax.set_ylabel("Total Sales ($)", fontsize=10, labelpad=8, color="#334155")
    ax.set_xlabel("Region", fontsize=10, labelpad=8, color="#334155")
    ax.set_ylim(0, max_val * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig
