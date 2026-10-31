from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
DATA_PATH = ROOT / "data" / "historical_forecasts.csv"


def load_data():
    return pd.read_csv(
        DATA_PATH,
        dtype={"stock_code": "string"},
        parse_dates=["sale_date"],
    )


def start_app():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=60)

    assert not app.exception, [
        error.message for error in app.exception
    ]

    return app


def displayed_metrics(app):
    return {
        metric.label: metric.value
        for metric in app.metric
    }


def metric_number(app, label):
    value = displayed_metrics(app)[label]
    return float(value.replace(" units", "").replace(",", ""))


def test_forecast_data_contract():
    data = load_data()

    assert set(data.columns) == {
        "stock_code",
        "sale_date",
        "baseline_forecast",
    }

    assert not data.isna().any().any()

    assert not data.duplicated(
        ["stock_code", "sale_date"]
    ).any()

    assert data["baseline_forecast"].ge(0).all()

    assert data["stock_code"].nunique() == 20

    assert data.groupby("sale_date").size().eq(20).all()

    assert data["sale_date"].min() == pd.Timestamp("2011-10-01")
    assert data["sale_date"].max() == pd.Timestamp("2011-12-08")


def test_app_loads():
    app = start_app()

    assert len(app.selectbox) == 1
    assert len(app.slider) == 1
    assert len(app.number_input) == 2

    assert set(displayed_metrics(app)) == {
        "Scenario demand",
        "Available stock",
        "Projected shortage",
        "Products with shortages",
    }


@pytest.mark.parametrize("change", [-100, 0, 20, 40])
def test_displayed_scenario_arithmetic(change):
    app = start_app()

    app.slider[0].set_value(change)
    app.number_input[0].set_value(50)
    app.number_input[1].set_value(10)
    app.run(timeout=60)

    assert not app.exception

    data = load_data()
    selected_date = app.selectbox[0].value

    forecasts = data.loc[
        data["sale_date"].dt.date.eq(selected_date),
        "baseline_forecast",
    ]

    scenario_demand = forecasts * (1 + change / 100)
    available_per_product = 60

    expected_demand = scenario_demand.sum()

    expected_shortage = (
        scenario_demand - available_per_product
    ).clip(lower=0).sum()

    expected_affected = int(
        scenario_demand.gt(available_per_product).sum()
    )

    metrics = displayed_metrics(app)

    assert metrics["Scenario demand"] == (
        f"{expected_demand:,.1f} units"
    )

    assert metrics["Available stock"] == (
        f"{available_per_product * len(forecasts):,.0f} units"
    )

    assert metrics["Projected shortage"] == (
        f"{expected_shortage:,.1f} units"
    )

    assert metrics["Products with shortages"] == (
        f"{expected_affected} / {len(forecasts)}"
    )


def test_more_demand_does_not_reduce_shortage():
    app = start_app()

    app.slider[0].set_value(0)
    app.run(timeout=60)
    assert not app.exception

    baseline_shortage = metric_number(
        app, "Projected shortage"
    )

    app.slider[0].set_value(40)
    app.run(timeout=60)
    assert not app.exception

    increased_shortage = metric_number(
        app, "Projected shortage"
    )

    assert increased_shortage >= baseline_shortage


def test_more_stock_does_not_increase_shortage():
    app = start_app()

    app.number_input[0].set_value(50)
    app.run(timeout=60)
    assert not app.exception

    original_shortage = metric_number(
        app, "Projected shortage"
    )

    app.number_input[0].set_value(100)
    app.run(timeout=60)
    assert not app.exception

    revised_shortage = metric_number(
        app, "Projected shortage"
    )

    assert revised_shortage <= original_shortage

