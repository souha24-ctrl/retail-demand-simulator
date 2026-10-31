
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRAIN_END = pd.Timestamp("2011-07-31")
VALIDATION_END = pd.Timestamp("2011-09-30")
MODEL_END = pd.Timestamp("2011-12-08")

COLUMN_MAPPING = {
    "Invoice": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

NUMERIC_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
]

FEATURES = ["stock_code"] + NUMERIC_FEATURES


def load_and_clean(workbook):
    frames = []

    with pd.ExcelFile(workbook) as excel:
        for sheet in excel.sheet_names:
            frame = pd.read_excel(excel, sheet_name=sheet)
            frame = frame.rename(columns=COLUMN_MAPPING)

            missing = set(COLUMN_MAPPING.values()) - set(frame.columns)
            if missing:
                raise ValueError(f"Missing columns in {sheet}: {missing}")

            frame["source_sheet"] = sheet
            frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)

    for column in ["invoice_no", "stock_code", "description", "country"]:
        raw[column] = raw[column].astype("string")

    raw["invoice_date"] = pd.to_datetime(
        raw["invoice_date"], errors="coerce"
    )

    for column in ["quantity", "unit_price", "customer_id"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    if raw[
        ["invoice_no", "stock_code", "invoice_date", "quantity", "unit_price"]
    ].isna().any().any():
        raise ValueError("Missing essential fields; inspect before continuing.")

    positive_pair = (
        raw["invoice_no"].eq("541431")
        & raw["stock_code"].eq("23166")
        & raw["customer_id"].eq(12346)
        & raw["quantity"].eq(74215)
        & raw["invoice_date"].eq(pd.Timestamp("2011-01-18 10:01:00"))
    )

    negative_pair = (
        raw["invoice_no"].eq("C541433")
        & raw["stock_code"].eq("23166")
        & raw["customer_id"].eq(12346)
        & raw["quantity"].eq(-74215)
        & raw["invoice_date"].eq(pd.Timestamp("2011-01-18 10:17:00"))
    )

    if positive_pair.sum() != 1 or negative_pair.sum() != 1:
        raise ValueError("The documented reversal pair was not uniquely found.")

    if (
        raw.loc[positive_pair, "unit_price"].iloc[0]
        != raw.loc[negative_pair, "unit_price"].iloc[0]
    ):
        raise ValueError("Reversal pair prices do not match.")

    audit = [{"step": "raw", "rows": len(raw)}]

    sales = raw.drop_duplicates().copy()
    audit.append({"step": "deduplicated", "rows": len(sales)})

    sales = sales.loc[
        ~sales["invoice_no"].str.upper().str.startswith("C")
    ].copy()
    audit.append({"step": "exclude_cancellations", "rows": len(sales)})

    sales = sales.loc[sales["quantity"].ge(0)].copy()
    audit.append({"step": "exclude_negative_quantities", "rows": len(sales)})

    sales = sales.loc[sales["unit_price"].gt(0)].copy()
    audit.append({"step": "exclude_nonpositive_prices", "rows": len(sales)})

    sales = sales.loc[~sales.index.isin(raw.index[positive_pair])].copy()
    audit.append({"step": "exclude_verified_reversal", "rows": len(sales)})

    sales["sale_date"] = sales["invoice_date"].dt.normalize()

    sales = sales.loc[
        sales["sale_date"].between("2009-12-01", MODEL_END)
    ].copy()

    audit.append({"step": "modeling_date_window", "rows": len(sales)})

    return sales, pd.DataFrame(audit)


def build_features(sales):
    training_sales = sales.loc[sales["sale_date"].le(TRAIN_END)]

    summary = (
        training_sales.groupby("stock_code")
        .agg(
            active_sales_days=("sale_date", "nunique"),
            first_observed_sale=("sale_date", "min"),
            last_training_sale=("sale_date", "max"),
            training_units=("quantity", "sum"),
        )
        .reset_index()
    )

    summary["calendar_span_days"] = (
        summary["last_training_sale"] - summary["first_observed_sale"]
    ).dt.days + 1

    selected = (
        summary.loc[
            summary["active_sales_days"].ge(30)
            & summary["calendar_span_days"].ge(180)
            & summary["training_units"].ge(100)
        ]
        .sort_values(
            ["training_units", "stock_code"],
            ascending=[False, True],
        )
        .head(20)
        .copy()
    )

    if len(selected) != 20:
        raise ValueError("Expected 20 eligible products.")

    observed = (
        sales.loc[sales["stock_code"].isin(selected["stock_code"])]
        .groupby(["stock_code", "sale_date"], as_index=False)
        .agg(daily_quantity=("quantity", "sum"))
    )

    calendars = [
        pd.DataFrame({
            "stock_code": row.stock_code,
            "sale_date": pd.date_range(row.first_observed_sale, MODEL_END),
        })
        for row in selected.itertuples(index=False)
    ]

    daily = pd.concat(calendars, ignore_index=True).merge(
        observed,
        on=["stock_code", "sale_date"],
        how="left",
        validate="one_to_one",
    )

    daily["daily_quantity"] = daily["daily_quantity"].fillna(0).astype("int64")
    daily = daily.sort_values(["stock_code", "sale_date"]).reset_index(drop=True)

    for lag in [1, 7, 14]:
        daily[f"lag_{lag}"] = (
            daily.groupby("stock_code")["daily_quantity"].shift(lag)
        )

    for window in [7, 28]:
        daily[f"rolling_mean_{window}"] = (
            daily.groupby("stock_code")["daily_quantity"]
            .transform(
                lambda values: values.shift(1)
                .rolling(window, min_periods=window)
                .mean()
            )
        )

    daily["previous_7day_median"] = (
        daily.groupby("stock_code")["daily_quantity"]
        .transform(
            lambda values: values.shift(1)
            .rolling(7, min_periods=7)
            .median()
        )
    )

    daily["day_of_week"] = daily["sale_date"].dt.dayofweek
    daily["month"] = daily["sale_date"].dt.month

    daily["split"] = "test"
    daily.loc[daily["sale_date"].le(VALIDATION_END), "split"] = "validation"
    daily.loc[daily["sale_date"].le(TRAIN_END), "split"] = "train"

    return daily.dropna(subset=NUMERIC_FEATURES).copy(), selected


def make_model():
    preprocessing = ColumnTransformer([
        (
            "product",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["stock_code"],
        ),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ])

    return Pipeline([
        ("preprocessor", preprocessing),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                criterion="absolute_error",
                max_depth=12,
                min_samples_leaf=5,
                max_features=1.0,
                random_state=42,
                n_jobs=2,
            ),
        ),
    ])


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE_percent": (
            100 * np.abs(actual - predicted).sum() / actual.sum()
            if actual.sum() > 0 else np.nan
        ),
        "mean_error_pred_minus_actual": (predicted - actual).mean(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the selected retail forecasting pipeline."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    args.output.mkdir(parents=True, exist_ok=True)

    sales, audit = load_and_clean(args.input)
    features, selected = build_features(sales)

    train = features.loc[features["split"].eq("train")]
    model = make_model()
    model.fit(train[FEATURES], train["daily_quantity"])

    results = []

    for split in ["validation", "test"]:
        evaluation = features.loc[features["split"].eq(split)].copy()
        evaluation["baseline_forecast"] = model.predict(evaluation[FEATURES])
        evaluation["zero_reference"] = 0.0

        for name, column in [
            ("Random Forest v2", "baseline_forecast"),
            ("Previous 7-day median", "previous_7day_median"),
            ("Always zero", "zero_reference"),
        ]:
            results.append({
                "split": split,
                "model": name,
                **metrics(evaluation["daily_quantity"], evaluation[column]),
            })

        evaluation[
            [
                "stock_code", "sale_date", "daily_quantity",
                "baseline_forecast", "previous_7day_median",
            ]
        ].to_csv(args.output / f"{split}_predictions.csv", index=False)

        if split == "test":
            evaluation[
                ["stock_code", "sale_date", "baseline_forecast"]
            ].to_csv(args.output / "historical_forecasts.csv", index=False)

    result_table = pd.DataFrame(results)
    result_table.to_csv(args.output / "metrics.csv", index=False)
    audit.to_csv(args.output / "cleaning_summary.csv", index=False)
    selected.to_csv(args.output / "selected_products.csv", index=False)

    features.groupby("split").size().to_csv(
        args.output / "split_row_counts.csv",
        header=["rows"],
    )

    joblib.dump(model, args.output / "selected_model.joblib")

    metadata = {
        "python": platform.python_version(),
        "packages": {
            package: version(package)
            for package in [
                "pandas", "numpy", "scikit-learn",
                "scipy", "openpyxl", "joblib",
            ]
        },
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "features": FEATURES,
        "model_parameters": model.named_steps["model"].get_params(),
        "training_end": str(TRAIN_END.date()),
        "validation_end": str(VALIDATION_END.date()),
        "test_end": str(MODEL_END.date()),
        "refitted_on_validation": False,
    }

    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(result_table.round(3).to_string(index=False))
    print("\nOutputs saved to:", args.output)


if __name__ == "__main__":
    main()
