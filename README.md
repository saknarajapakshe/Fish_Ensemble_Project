# 🐟 Fair Retail Fish Price Checker: Ensemble Learning on Sri Lankan Fish Prices
Rajapakshe R.A.R.S - 224223N

**Question:** given this week's Peliyagoda **wholesale** price of a fish species, what **retail** price should a Colombo shopper expect?
The project compares **bagging, boosting, stacking and an RF + ANN hybrid** against simple baselines, and explains why they behave differently.



## Data
Ministry of Fisheries weekly fish price reports (Rs/kg), Jan 2019 – Apr 2026: 31 species, wholesale (Peliyagoda) and retail (Colombo markets).
Source: the original weekly Excel reports archived in [nuuuwan/lk_fisheries](https://github.com/nuuuwan/lk_fisheries) (listed as dataset 024 in [nuuuwan/lk_datasets](https://github.com/nuuuwan/lk_datasets)).
The notebook **downloads all 277 reports at a fixed commit** (`3847e3b…`, snapshot of 23 Nov 2026) and combines them itself, so no pre-made data file is needed. Internet access is required.

## Results (test: 2024 – Apr 2026)
| Model | Family | Test MAPE |
|---|---|---|
| Hybrid RF + ANN | Hybrid | **6.88%** |
| Bagging (decision trees) | Bagging | **6.89%** |
| LightGBM | Boosting | 6.98% |
| Random forest | Bagging | 7.17% |
| Stacking | Stacking | 7.25% |
| CatBoost | Boosting | 7.35% |
| GBM | Boosting | 7.69% |
| Ridge | Baseline | 8.20% |
| Decision tree | Single tree | 9.50% |
| Species median markup | Baseline | 9.60% |
| AdaBoost | Boosting | 14.46% |

## Files
| File | What it is |
|---|---|
| `fish_ensemble.ipynb` | Full pipeline: download reports, combine, clean, features, models, evaluation, why-analysis (runs in Google Colab) |
| `app.py` | Streamlit dashboard |
| `models/model_bundle.joblib` | Trained models used by the dashboard |
| `results/*.csv` | Metrics and analysis tables written by the notebook (including the combined dataset `fish_prices_long.csv`) |
| `interpretation.md` | Findings text shown in the dashboard (student-written) |
| `AI_DISCLOSURE.md`, `prompts.md` | Generative AI disclosure |

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```


