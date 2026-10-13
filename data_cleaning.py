"""
data_cleaning.py
Comprehensive data cleaning and preprocessing pipeline.
Handles:
- Data inspection and profiling
- Inconsistent types, currency formatting, mixed date parsing
- Missing value imputation and removal strategies
- Duplicate detection and removal
- Outlier detection and treatment using IQR method
- Invalid record filtering
- Before-and-after cleaning comparison reports
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects raw or cleaned dataset and produces high-level metrics and column statistics.
    """
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0
    duplicates = int(df.duplicated().sum())

    col_stats = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_ratio = (null_count / total_rows * 100) if total_rows > 0 else 0.0
        n_unique = int(df[col].nunique(dropna=True))
        dtype = str(df[col].dtype)
        sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None"
        
        col_stats.append({
            "Column": col,
            "Type": dtype,
            "Missing Values": null_count,
            "Missing %": round(null_ratio, 2),
            "Unique Values": n_unique,
            "Sample Value": sample_val
        })

    col_stats_df = pd.DataFrame(col_stats)

    return {
        "rows": total_rows,
        "cols": total_cols,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_pct, 2),
        "duplicates": duplicates,
        "col_stats_df": col_stats_df
    }


def clean_currency_and_numbers(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    Standardizes numeric columns by stripping currency symbols ($), commas, and spaces,
    then converts to numeric float/int.
    """
    df = df.copy()
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[\$,]", "", regex=True)
                    .str.strip()
                    .replace({"": np.nan, "None": np.nan, "nan": np.nan})
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def standardize_text_and_dates(
    df: pd.DataFrame,
    text_cols: List[str],
    date_cols: List[str]
) -> pd.DataFrame:
    """
    Standardizes text columns (stripping whitespace and formatting title case)
    and parses mixed-format date columns safely into datetime objects.
    """
    df = df.copy()
    
    # Standardize text
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.strip().title() if isinstance(x, str) and x.strip() else (np.nan if pd.isna(x) or str(x).strip() == "" else x)
            )

    # Standardize dates
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    return df


def filter_invalid_records(
    df: pd.DataFrame,
    positive_cols: List[str]
) -> Tuple[pd.DataFrame, int]:
    """
    Removes records with physically impossible/negative numbers (e.g. Quantity <= 0).
    Returns (cleaned_df, count_removed).
    """
    df = df.copy()
    initial_rows = len(df)
    
    mask = pd.Series(True, index=df.index)
    for col in positive_cols:
        if col in df.columns:
            valid_col_mask = df[col].isna() | (df[col] > 0)
            mask = mask & valid_col_mask
            
    filtered_df = df[mask].reset_index(drop=True)
    removed_count = initial_rows - len(filtered_df)
    return filtered_df, removed_count


def reconcile_sales_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconciles 'Total Sales' = 'Quantity' * 'Unit Price' if present, ensuring accuracy
    and filling missing total sales when quantity and unit price exist.
    """
    df = df.copy()
    if "Quantity" in df.columns and "Unit Price" in df.columns:
        calculated_sales = df["Quantity"] * df["Unit Price"]
        if "Total Sales" in df.columns:
            # Fill missing or update inconsistent Total Sales
            df["Total Sales"] = df["Total Sales"].fillna(calculated_sales)
            # Reconcile any mismatch larger than 0.05
            mismatch = (df["Total Sales"] - calculated_sales).abs() > 0.05
            df.loc[mismatch & df["Quantity"].notna() & df["Unit Price"].notna(), "Total Sales"] = calculated_sales
        else:
            df["Total Sales"] = calculated_sales
    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "unknown"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handles missing values based on specified strategies.
    numeric_strategy: 'median', 'mean', or 'drop'
    categorical_strategy: 'mode', 'unknown', or 'drop'
    """
    df = df.copy()
    missing_actions = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_strategy == "drop":
        before_drop = len(df)
        df = df.dropna(subset=numeric_cols)
        missing_actions["numeric"] = f"Dropped {before_drop - len(df)} rows with missing numeric values"
    else:
        for col in numeric_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                fill_val = df[col].median() if numeric_strategy == "median" else df[col].mean()
                fill_val = round(fill_val, 2)
                df[col] = df[col].fillna(fill_val)
                missing_actions[col] = f"Imputed {n_missing} missing values with {numeric_strategy} ({fill_val})"

    if categorical_strategy == "drop":
        before_drop = len(df)
        df = df.dropna(subset=categorical_cols)
        missing_actions["categorical"] = f"Dropped {before_drop - len(df)} rows with missing categorical values"
    else:
        for col in categorical_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                if categorical_strategy == "mode":
                    mode_series = df[col].mode()
                    fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                else:
                    fill_val = "Unknown"
                df[col] = df[col].fillna(fill_val)
                missing_actions[col] = f"Filled {n_missing} missing values with '{fill_val}'"

    # For date columns, drop rows where date could not be parsed if any
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    for col in date_cols:
        n_missing_date = int(df[col].isna().sum())
        if n_missing_date > 0:
            df = df.dropna(subset=[col])
            missing_actions[col] = f"Dropped {n_missing_date} rows with unparseable date values"

    return df.reset_index(drop=True), missing_actions


def remove_duplicate_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Identifies and removes duplicate records.
    Returns (cleaned_df, duplicates_removed_count).
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_count - len(df_clean)
    return df_clean, duplicates_removed


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    factor: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Detects outliers using the Interquartile Range (IQR) method:
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return pd.DataFrame(), {}

    series = df[column].dropna()
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = float(q1 - factor * iqr)
    upper_bound = float(q3 + factor * iqr)

    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers_df = df[outliers_mask]

    bounds = {
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outlier_count": int(len(outliers_df))
    }
    return outliers_df, bounds


def handle_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "clip",
    factor: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Handles outliers across specified numerical columns.
    method:
      - 'clip': Caps outliers to [lower_bound, upper_bound]
      - 'remove': Drops rows where value is outside bounds
      - 'keep': Retains outliers as is
    """
    df = df.copy()
    outlier_report = {}

    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            _, bounds = detect_outliers_iqr(df, col, factor=factor)
            count = bounds.get("outlier_count", 0)

            if method == "clip" and count > 0:
                df[col] = df[col].clip(lower=bounds["lower_bound"], upper=bounds["upper_bound"])
                action = f"Capped {count} values to [{bounds['lower_bound']}, {bounds['upper_bound']}]"
            elif method == "remove" and count > 0:
                mask = (df[col] >= bounds["lower_bound"]) & (df[col] <= bounds["upper_bound"]) | df[col].isna()
                df = df[mask].reset_index(drop=True)
                action = f"Removed {count} outlier records outside [{bounds['lower_bound']}, {bounds['upper_bound']}]"
            else:
                action = f"Retained {count} outliers without modification"

            outlier_report[col] = {
                **bounds,
                "action": action,
                "method_used": method
            }

    return df, outlier_report


def run_full_cleaning_pipeline(
    raw_df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "unknown",
    outlier_method: str = "clip",
    outlier_columns: Optional[List[str]] = None,
    iqr_factor: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the end-to-end cleaning workflow and returns (cleaned_df, summary_report).
    """
    df = raw_df.copy()
    initial_inspection = inspect_dataset(df)
    
    # 1. Clean column names
    df.columns = [c.strip() for c in df.columns]

    # 2. Standardize numbers and currency
    num_candidates = ["Quantity", "Unit Price", "Total Sales"]
    df = clean_currency_and_numbers(df, num_candidates)

    # 3. Standardize text and dates
    text_candidates = ["Product Name", "Category", "Region", "Customer Name"]
    date_candidates = ["Order Date"]
    df = standardize_text_and_dates(df, text_candidates, date_candidates)

    # 4. Remove invalid records (e.g. Quantity <= 0 or Unit Price <= 0)
    positive_cols = [c for c in ["Quantity", "Unit Price"] if c in df.columns]
    df, invalid_count = filter_invalid_records(df, positive_cols)

    # 5. Handle missing values
    df, missing_actions = handle_missing_values(
        df,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy
    )

    # 6. Reconcile sales calculations
    df = reconcile_sales_column(df)

    # 7. Remove duplicate records
    df, duplicates_removed = remove_duplicate_records(df)

    # 8. Outliers
    if outlier_columns is None:
        outlier_columns = [c for c in ["Quantity", "Unit Price", "Total Sales"] if c in df.columns]
    
    df, outlier_report = handle_outliers(
        df,
        columns=outlier_columns,
        method=outlier_method,
        factor=iqr_factor
    )

    # Final inspection
    final_inspection = inspect_dataset(df)

    summary_report = {
        "initial_rows": initial_inspection["rows"],
        "final_rows": final_inspection["rows"],
        "rows_removed": initial_inspection["rows"] - final_inspection["rows"],
        "initial_missing_cells": initial_inspection["missing_cells"],
        "final_missing_cells": final_inspection["missing_cells"],
        "initial_duplicates": initial_inspection["duplicates"],
        "final_duplicates": final_inspection["duplicates"],
        "invalid_records_removed": invalid_count,
        "duplicates_removed": duplicates_removed,
        "missing_actions": missing_actions,
        "outlier_report": outlier_report,
        "raw_col_stats": initial_inspection["col_stats_df"],
        "clean_col_stats": final_inspection["col_stats_df"],
    }

    return df, summary_report
