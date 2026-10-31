# Retail Demand & Resource Allocation Simulator

A retail sales-forecasting project with an interactive dashboard for exploring
potential stock shortages under hypothetical demand and inventory scenarios.

[Open the live dashboard](https://souha-retail-demand-simulator.streamlit.app/)

[Explore the research notebook](/retail_demand_simulator.ipynb)

## Research question

Can historical retail transactions support one-day-ahead product-sales forecasting
and help users explore stock-shortage scenarios?

The project combines data-quality investigation, chronological model evaluation,
and a lightweight decision-support interface.

## What the dashboard does

Users can select:

- A historical forecast date.
- An assumed percentage change in demand.
- Starting stock per product.
- Replenishment per product.

The dashboard updates:

- Scenario demand.
- Available inventory.
- Projected shortages and surpluses.
- Number of products with projected shortages.
- Product-level charts and a downloadable scenario table.

The current interface applies the same stock and replenishment settings to every
product. Products remain separate: surplus stock of one product does not offset
another product's shortage.

## Architecture

```mermaid
flowchart LR
    A[UCI transaction workbook] --> B[Audit and documented cleaning]
    B --> C[Daily product sales]
    C --> D[Training-only product selection]
    D --> E[Past-only forecasting features]
    E --> F[Baselines and Random Forest]
    F --> G[Chronological evaluation]
    G --> H[Saved historical forecasts]
    H --> I[Streamlit scenario dashboard]
    I --> J[Projected shortages and CSV export]
```

Model training takes place in Google Colab.
The deployed dashboard reads saved predictions; it does not retrain the model
or load the original transaction workbook.

## Dataset

Source: UCI Online Retail II.

The original workbook contains 1,067,371 transaction records spanning
December 2009–December 2011.

Source page:
https://archive.ics.uci.edu/dataset/502/online+retail+ii

Dataset citation:
Chen, D. (2012). Online Retail II [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D

The transaction data does not provide historical stock levels or supplier
delivery schedules. Inventory values in the dashboard are scenario assumptions.

## Data preparation

The workflow:

1. Loads both annual workbook sheets.
2. Standardizes column names and text types.
3. Audits missing values, duplicate rows, cancellations, quantities, and prices.
4. Removes exact duplicates and excludes cancellation rows, negative quantities,
   and non-positive prices from the positive-sales table.
5. Retains missing customer identifiers because product-level aggregation does
   not require customer identity.
6. Applies a documented exclusion for an inspected same-day reversed order.
7. Aggregates transactions to product-date sales quantities.

### Investigated reversal

An unusually large daily spike was traced to a 74,215-unit order followed
16 minutes later by an equal negative transaction for the same product,
customer, and unit price.

The positive order was excluded through an explicit, case-specific rule.
Large orders were not removed automatically.

This is not a comprehensive matching system for all cancellations and returns.

### Target definition

The target is positive recorded sales quantity after the documented exclusions.

It is not:
- Net sales after all returns.
- Unconstrained customer demand.
- Confirmed inventory consumption.

Dates without recorded transactions are assigned zero observed sales after
each product's first observed training-period sale. Such zeros do not establish
zero demand, product availability, or store closure.

## Forecasting protocol

Twenty products are selected using training-period information only.

Eligibility requires:
- At least 30 active sales days.
- At least 180 calendar days between first and last training-period sale.
- At least 100 training-period units.

Eligible products are ranked by training-period sales volume.

| Split | Target dates | Feature-ready product-day rows |
|---|---|---:|
| Training | 29 Dec 2009–31 Jul 2011 | 11,288 |
| Validation | 1 Aug–30 Sep 2011 | 1,220 |
| Test | 1 Oct–8 Dec 2011 | 1,380 |

The earliest raw training observations begin on 1 December 2009.
The first 28 observations per product are excluded from model fitting because
the features require historical context.

The final extract date, 9 December 2011, is excluded because it may be incomplete.

### Features

- Product identity, one-hot encoded.
- Sales lags of 1, 7, and 14 days.
- Previous 7-day and 28-day mean sales.
- Target-day weekday and month.

Rolling statistics are shifted to exclude the target day's sales.
Same-day sales-derived predictors and backward-filled prices are not used.

Evaluation follows a rolling one-day-ahead protocol: each target date may use
actual observations from earlier dates, including earlier dates within the
evaluation period.

This is not a multi-day forecast issued from a single origin.

## Model selection

The project compares:

- Always-zero reference.
- Previous-week same-day sales.
- Previous-seven-day mean.
- Previous-seven-day median.
- Random Forest with squared-error criterion.
- Random Forest with absolute-error criterion.

The absolute-error Random Forest is selected using validation MAE.
It is not refitted on validation data before the reported test evaluation.

Selected configuration:

```python
RandomForestRegressor(
    n_estimators=200,
    criterion="absolute_error",
    max_depth=12,
    min_samples_leaf=5,
    max_features=1.0,
    random_state=42,
    n_jobs=2,
)
```

## Held-out test results

| Method | MAE ↓ | RMSE ↓ | WAPE ↓ |
|---|---:|---:|---:|
| Random Forest: absolute error | 65.535 | 217.652 | 75.706% |
| Previous-seven-day median | 74.264 | 223.576 | 85.791% |
| Always-zero reference | 86.564 | 246.750 | 100.000% |

The selected forest reduces test MAE by approximately 11.8% relative to
the previous-seven-day median.

MAE is measured in units per product-day.
WAPE is total absolute error divided by total actual units; it is not
percentage accuracy.

### Directional error

- Actual test sales: 119,459 units.
- Total forest forecasts: 76,074.708 units.
- Product-day underforecast units: 66,911.052.
- Product-day overforecast units: 23,526.760.
- Mean forecast error: −31.438 units per product-day.

The model outperforms the reported references but substantially underpredicts
aggregate sales and misses major peaks.

Forecast underprediction is not an observed stockout: actual inventory levels
are unavailable.

## Scenario calculations

For each product:

```text
scenario demand = forecast × (1 + demand change percentage / 100)

available units = starting stock + replenishment

projected shortage = max(scenario demand − available units, 0)

projected surplus = max(available units − scenario demand, 0)
```

Replenishment is assumed available before demand must be served.

The demand-change setting is a deterministic assumption. It does not estimate
a causal effect, a confidence interval, or a shortage probability.

Fractional quantities are retained for analytical display rather than presented
as rounded purchase recommendations.

## Run the dashboard

The hosted application requires no local installation:

https://souha-retail-demand-simulator.streamlit.app/

Optional local execution from the repository root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard requires `data/historical_forecasts.csv`.
It does not require the trained model artifact or raw customer-level records.

For the research workflow, open the notebook in Google Colab and follow its
setup and data-acquisition sections. The root `requirements.txt` is for the
dashboard, not the full training environment.

## Repository structure

```text
retail-demand-simulator/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── retail_demand_simulator.ipynb
├── data/
│   ├── README.md
│   └── historical_forecasts.csv
└── tests/
    └── test_dashboard.py
```

## Limitations

- Recorded sales are an imperfect proxy for demand.
- Historical inventory, stockouts, and delivery timing are unavailable.
- The selected products emphasize historically high-volume items.
- Zero-sales dates cannot reliably distinguish no demand from unavailability.
- Cancellation matching is incomplete.
- Exact-duplicate removal is a documented assumption: identical lines can
  sometimes represent legitimate repeated transactions.
- Model comparison uses one chronological validation period and one test period.
- No statistical significance or uncertainty intervals are claimed.
- Large sales spikes remain difficult to forecast.
- Forecast underprediction can understate scenario shortages.
- The dashboard uses historical predictions, not live operational inputs.
- The app does not optimize allocation or model inventory across multiple days.

## Future work

- Separate training and cleaning into reusable scripts.
- Expand automated tests to cover the training pipeline, invalid inputs,
and additional interface behavior.
- Introduce product-specific inventory inputs.
- Evaluate rolling-origin backtests across multiple periods.
- Investigate quantile forecasts and asymmetric shortage costs.
- Add multi-day inventory balances and delivery lead times.
- Implement replenishment recommendations under an explicit budget constraint.

## Research takeaway

A lightweight machine-learning model can improve on simple forecasting references
while remaining insufficient as an autonomous inventory-planning system.

The contribution is a transparent workflow connecting data-quality investigation,
chronological evaluation, and explicit operational assumptions. It is not a claim that
point forecasts alone solve resource allocation.

## Automated tests

The repository includes eight dashboard test cases using pytest and
Streamlit AppTest.

The tests check:
- Forecast CSV columns, missing values, product coverage, and date boundaries.
- Successful application execution.
- Displayed scenario arithmetic at −100%, 0%, +20%, and +40% demand.
- Increasing demand does not reduce projected shortages.
- Increasing stock does not increase projected shortages.

Install the development dependencies and run from the repository root:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/test_dashboard.py -q
```

The current test suite passed all eight cases in the development environment.

These tests validate dashboard behavior, not forecasting accuracy.
They do not constitute a comprehensive security, accessibility, or browser-compatibility audit.
Forecast accuracy is reported separately using the chronological evaluation.
