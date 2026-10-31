# Dashboard data

`historical_forecasts.csv` contains precomputed one-day-ahead sales forecasts
for 20 selected products during the held-out historical evaluation period.

Columns:
- `stock_code`: product identifier.
- `sale_date`: historical forecast target date.
- `baseline_forecast`: Random Forest sales forecast in units.

The file contains derived forecasts only, not customer-level transactions.

Source dataset:
UCI Online Retail II, Chen (2012).
DOI: 10.24432/C5CG6D
