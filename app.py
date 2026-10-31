
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Retail Demand Simulator",
    page_icon="📦",
    layout="wide",
)

DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "historical_forecasts.csv"
)


@st.cache_data
def load_forecasts():
    data = pd.read_csv(
        DATA_PATH,
        dtype={"stock_code": "string"},
        parse_dates=["sale_date"],
    )

    required = {"stock_code", "sale_date", "baseline_forecast"}

    if not required.issubset(data.columns):
        raise ValueError("Forecast file is missing required columns.")

    if data[list(required)].isna().any().any():
        raise ValueError("Forecast file contains missing values.")

    if data.duplicated(["stock_code", "sale_date"]).any():
        raise ValueError("Forecast file contains duplicate product-date rows.")

    if not data["baseline_forecast"].ge(0).all():
        raise ValueError("Forecast quantities must be non-negative.")

    return data.sort_values(["sale_date", "stock_code"])


def simulate(data, selected_date, demand_change, stock, replenishment):
    result = data.loc[
        data["sale_date"].dt.date.eq(selected_date)
    ].copy()

    result["scenario_demand"] = (
        result["baseline_forecast"] * (1 + demand_change / 100)
    )

    result["starting_stock"] = stock
    result["replenishment"] = replenishment
    result["available_units"] = stock + replenishment

    result["projected_shortage"] = (
        result["scenario_demand"] - result["available_units"]
    ).clip(lower=0)

    result["projected_surplus"] = (
        result["available_units"] - result["scenario_demand"]
    ).clip(lower=0)

    return result.sort_values(
        ["projected_shortage", "stock_code"],
        ascending=[False, True],
    )


st.title("Retail Demand & Resource Allocation Simulator")

st.caption(
    "Historical scenario explorer | "
    "20 products | One-day-ahead Random Forest forecasts"
)

st.warning(
    "This is a research demonstration, not a live inventory system. "
    "Stock and replenishment are hypothetical inputs. "
    "The model underforecast total sales in held-out testing; "
    "scenario shortages are not observed stockouts or shortage probabilities."
)

forecasts = load_forecasts()
available_dates = sorted(forecasts["sale_date"].dt.date.unique())

preferred_date = pd.Timestamp("2011-11-01").date()
default_index = (
    available_dates.index(preferred_date)
    if preferred_date in available_dates
    else 0
)

with st.sidebar:
    st.header("Scenario assumptions")

    selected_date = st.selectbox(
        "Historical forecast date",
        options=available_dates,
        index=default_index,
        format_func=lambda value: value.strftime("%d %B %Y"),
    )

    demand_change = st.slider(
        "Assumed demand change (%)",
        min_value=-100,
        max_value=100,
        value=20,
        step=5,
    )

    stock = st.number_input(
        "Starting stock per product (units)",
        min_value=0,
        value=50,
        step=10,
    )

    replenishment = st.number_input(
        "Replenishment per product (units)",
        min_value=0,
        value=10,
        step=10,
    )

    st.caption(
        "The same stock settings apply to all products. "
        "Replenishment is assumed available before demand is served."
    )

scenario = simulate(
    forecasts,
    selected_date,
    demand_change,
    stock,
    replenishment,
)

st.subheader(
    f"Scenario for {selected_date.strftime('%d %B %Y')}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Scenario demand",
    f"{scenario['scenario_demand'].sum():,.1f} units",
)

c2.metric(
    "Available stock",
    f"{scenario['available_units'].sum():,.0f} units",
)

c3.metric(
    "Projected shortage",
    f"{scenario['projected_shortage'].sum():,.1f} units",
)

c4.metric(
    "Products with shortages",
    f"{int(scenario['projected_shortage'].gt(0).sum())}"
    f" / {len(scenario)}",
)

st.caption(
    f"Baseline forecast: {scenario['baseline_forecast'].sum():,.1f} units. "
    f"Projected surplus: {scenario['projected_surplus'].sum():,.1f} units. "
    "Shortages and surpluses are calculated separately for each product."
)

st.subheader("Projected shortage by product")

shortage_chart = scenario[
    ["stock_code", "projected_shortage"]
].set_index("stock_code")

st.bar_chart(shortage_chart)

st.subheader("Forecast, scenario demand, and available stock")

comparison_chart = scenario[
    [
        "stock_code",
        "baseline_forecast",
        "scenario_demand",
        "available_units",
    ]
].set_index("stock_code")

st.bar_chart(comparison_chart)

st.subheader("Product-level scenario results")

st.dataframe(
    scenario.drop(columns=["sale_date"]).round(2),
    hide_index=True,
    use_container_width=True,
)

export = scenario.copy()
export["demand_change_percent"] = demand_change

st.download_button(
    label="Download scenario as CSV",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name=f"retail_scenario_{selected_date.isoformat()}.csv",
    mime="text/csv",
)

with st.expander("Methodology and limitations"):
    st.markdown("""
- Forecasts were generated using a Random Forest with an absolute-error criterion.
- Products were selected using training-period data only.
- Evaluation used rolling one-day-ahead predictions.
- Held-out test MAE: 65.535 units per product-day.
- Held-out test WAPE: 75.706%.
- Total test sales: 119,459 units; total forecasts: approximately 76,075 units.
- Recorded sales are a proxy for demand.
- Inventory and replenishment are user assumptions, not historical observations.
- Demand adjustments are deterministic scenarios, not causal estimates.
- This app does not optimize replenishment, model delivery delays, or estimate shortage probabilities.
- Fractional quantities are retained for analytical display; they are not rounded purchase recommendations.
""")

st.caption(
    "Data provenance: UCI Online Retail II, Chen (2012), "
    "DOI: 10.24432/C5CG6D. "
    "Dashboard data contains derived forecasts only."
)
