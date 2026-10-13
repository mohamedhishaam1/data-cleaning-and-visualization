"""
data_analysis.py
Statistical and business analysis module.
Computes:
- Core KPI metrics (Revenue, Orders, Avg Order Value, Total Items Sold)
- Product performance (Best-selling products by sales & volume)
- Category-level sales and revenue share
- Regional performance breakdown
- Monthly sales trends and MoM growth rates
- Customer statistics
- Automated executive summary and business insights
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes high-level business KPIs.
    """
    if df.empty:
        return {
            "total_sales": 0.0,
            "avg_sales": 0.0,
            "total_orders": 0,
            "total_quantity": 0,
            "avg_order_value": 0.0,
            "avg_unit_price": 0.0,
            "unique_customers": 0,
            "unique_products": 0
        }

    total_sales = float(df["Total Sales"].sum()) if "Total Sales" in df.columns else 0.0
    total_orders = int(df["Order ID"].nunique()) if "Order ID" in df.columns else len(df)
    total_quantity = int(df["Quantity"].sum()) if "Quantity" in df.columns else 0
    avg_sales = float(df["Total Sales"].mean()) if "Total Sales" in df.columns else 0.0
    avg_order_value = float(total_sales / total_orders) if total_orders > 0 else 0.0
    avg_unit_price = float(df["Unit Price"].mean()) if "Unit Price" in df.columns else 0.0
    unique_customers = int(df["Customer Name"].nunique()) if "Customer Name" in df.columns else 0
    unique_products = int(df["Product Name"].nunique()) if "Product Name" in df.columns else 0

    return {
        "total_sales": round(total_sales, 2),
        "avg_sales": round(avg_sales, 2),
        "total_orders": total_orders,
        "total_quantity": total_quantity,
        "avg_order_value": round(avg_order_value, 2),
        "avg_unit_price": round(avg_unit_price, 2),
        "unique_customers": unique_customers,
        "unique_products": unique_products
    }


def analyze_top_products(df: pd.DataFrame, top_n: int = 10, by: str = "Total Sales") -> pd.DataFrame:
    """
    Returns the best-performing products ranked by sales or quantity.
    """
    if "Product Name" not in df.columns:
        return pd.DataFrame()

    agg_dict = {}
    if "Total Sales" in df.columns:
        agg_dict["Total Sales"] = "sum"
    if "Quantity" in df.columns:
        agg_dict["Quantity"] = "sum"
    if "Order ID" in df.columns:
        agg_dict["Order ID"] = "count"

    grouped = df.groupby(["Product Name", "Category"], as_index=False).agg(agg_dict)
    if "Order ID" in grouped.columns:
        grouped = grouped.rename(columns={"Order ID": "Order Count"})

    sort_col = by if by in grouped.columns else ("Total Sales" if "Total Sales" in grouped.columns else "Quantity")
    ranked = grouped.sort_values(by=sort_col, ascending=False).head(top_n).reset_index(drop=True)
    
    if "Total Sales" in ranked.columns:
        ranked["Total Sales"] = ranked["Total Sales"].round(2)
        total_rev = df["Total Sales"].sum()
        ranked["Revenue Share %"] = (ranked["Total Sales"] / total_rev * 100).round(2) if total_rev > 0 else 0.0

    return ranked


def analyze_category_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes sales volume, revenue, and average transaction by Category.
    """
    if "Category" not in df.columns:
        return pd.DataFrame()

    agg_dict = {}
    if "Total Sales" in df.columns:
        agg_dict["Total Sales"] = ["sum", "mean"]
    if "Quantity" in df.columns:
        agg_dict["Quantity"] = "sum"
    if "Order ID" in df.columns:
        agg_dict["Order ID"] = "count"

    grouped = df.groupby("Category").agg(agg_dict)
    grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
    
    rename_map = {
        "Total Sales_sum": "Total Revenue",
        "Total Sales_mean": "Avg Ticket Size",
        "Quantity_sum": "Units Sold",
        "Order ID_count": "Order Count"
    }
    grouped = grouped.rename(columns=rename_map).reset_index()

    if "Total Revenue" in grouped.columns:
        total_rev = grouped["Total Revenue"].sum()
        grouped["Revenue Share %"] = (grouped["Total Revenue"] / total_rev * 100).round(2) if total_rev > 0 else 0.0
        grouped["Total Revenue"] = grouped["Total Revenue"].round(2)
    if "Avg Ticket Size" in grouped.columns:
        grouped["Avg Ticket Size"] = grouped["Avg Ticket Size"].round(2)

    return grouped.sort_values(by="Total Revenue", ascending=False).reset_index(drop=True)


def analyze_regional_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes regional metrics including sales, average transaction value, and order count.
    """
    if "Region" not in df.columns:
        return pd.DataFrame()

    agg_dict = {}
    if "Total Sales" in df.columns:
        agg_dict["Total Sales"] = ["sum", "mean"]
    if "Quantity" in df.columns:
        agg_dict["Quantity"] = "sum"
    if "Order ID" in df.columns:
        agg_dict["Order ID"] = "count"

    grouped = df.groupby("Region").agg(agg_dict)
    grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
    
    rename_map = {
        "Total Sales_sum": "Total Sales",
        "Total Sales_mean": "Average Order Value",
        "Quantity_sum": "Quantity Sold",
        "Order ID_count": "Order Count"
    }
    grouped = grouped.rename(columns=rename_map).reset_index()

    if "Total Sales" in grouped.columns:
        total_rev = grouped["Total Sales"].sum()
        grouped["Sales Share %"] = (grouped["Total Sales"] / total_rev * 100).round(2) if total_rev > 0 else 0.0
        grouped["Total Sales"] = grouped["Total Sales"].round(2)
    if "Average Order Value" in grouped.columns:
        grouped["Average Order Value"] = grouped["Average Order Value"].round(2)

    return grouped.sort_values(by="Total Sales", ascending=False).reset_index(drop=True)


def analyze_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes monthly sales trends, order volumes, and month-over-month (MoM) growth rates.
    """
    if "Order Date" not in df.columns or "Total Sales" not in df.columns:
        return pd.DataFrame()

    df_time = df.copy()
    df_time = df_time.dropna(subset=["Order Date"])
    df_time["Order Date"] = pd.to_datetime(df_time["Order Date"], errors="coerce")
    df_time = df_time.dropna(subset=["Order Date"])

    if df_time.empty:
        return pd.DataFrame()

    df_time["YearMonth"] = df_time["Order Date"].dt.to_period("M")
    
    monthly = df_time.groupby("YearMonth").agg(
        Total_Sales=("Total Sales", "sum"),
        Order_Count=("Order ID", "count") if "Order ID" in df_time.columns else ("Total Sales", "count"),
        Units_Sold=("Quantity", "sum") if "Quantity" in df_time.columns else ("Total Sales", "count"),
        Avg_Order_Value=("Total Sales", "mean")
    ).reset_index()

    monthly["Month_Str"] = monthly["YearMonth"].astype(str)
    monthly["Total_Sales"] = monthly["Total_Sales"].round(2)
    monthly["Avg_Order_Value"] = monthly["Avg_Order_Value"].round(2)
    monthly["MoM_Growth_%"] = (monthly["Total_Sales"].pct_change() * 100).round(2).fillna(0.0)

    return monthly


def generate_business_insights(df: pd.DataFrame) -> List[str]:
    """
    Generates actionable executive summary insights from cleaned data.
    """
    insights = []
    if df.empty:
        return ["No data available to formulate insights."]

    kpis = calculate_kpis(df)
    insights.append(
        f"**Revenue Overview:** Total sales reached **${kpis['total_sales']:,.2f}** across **{kpis['total_orders']} orders**, yielding an average order value (AOV) of **${kpis['avg_order_value']:,.2f}**."
    )

    cat_df = analyze_category_sales(df)
    if not cat_df.empty:
        top_cat = cat_df.iloc[0]
        insights.append(
            f"**Category Dominance:** **{top_cat['Category']}** is the highest revenue contributor at **${top_cat['Total Revenue']:,.2f}** ({top_cat['Revenue Share %']}% of total revenue)."
        )

    prod_df = analyze_top_products(df, top_n=1)
    if not prod_df.empty:
        top_prod = prod_df.iloc[0]
        insights.append(
            f"**Star Product:** **{top_prod['Product Name']}** leads sales with **${top_prod['Total Sales']:,.2f}** ({top_prod['Quantity']} units sold)."
        )

    reg_df = analyze_regional_sales(df)
    if not reg_df.empty:
        top_reg = reg_df.iloc[0]
        insights.append(
            f"**Top Performing Region:** The **{top_reg['Region']}** region generated the most revenue (**${top_reg['Total Sales']:,.2f}**, {top_reg['Sales Share %']}% share)."
        )

    trends_df = analyze_monthly_trends(df)
    if not trends_df.empty and len(trends_df) > 1:
        peak_month = trends_df.loc[trends_df["Total_Sales"].idxmax()]
        insights.append(
            f"**Peak Performance Period:** Highest monthly sales occurred in **{peak_month['Month_Str']}** with **${peak_month['Total_Sales']:,.2f}** in revenue."
        )

    return insights
