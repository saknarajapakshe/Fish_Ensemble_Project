import os, re
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fair Fish Price Checker", page_icon="🐟", layout="wide")


@st.cache_data
def load_tables():
    names = ["metrics_test", "ntrees_curve", "stack_weights", "importance", "species_error",
             "level_vs_markup", "bias_variance", "app_models_test"]
    t = {n: pd.read_csv(f"results/{n}.csv") for n in names}
    t["corr"] = pd.read_csv("results/residual_corr.csv", index_col=0)
    t["preds"] = pd.read_csv("results/predictions_test.csv", parse_dates=["date"])
    t["wide"] = pd.read_csv("results/wide_prices.csv", parse_dates=["date"])
    return t


@st.cache_resource
def load_bundle():
    return joblib.load("models/model_bundle.joblib")


T = load_tables()
B = load_bundle()
COLORS = {"Baseline": "#9e9e9e", "Single tree": "#424242", "Bagging": "#2e7d32",
          "Boosting": "#ef6c00", "Stacking": "#6a1b9a", "Hybrid": "#1565c0"}


def show_file(path):
    if os.path.exists(path):
        st.markdown(open(path, encoding="utf-8").read())


st.title("Fair Retail Fish Price Checker")
st.caption("Ensemble learning on Sri Lankan weekly fish prices (Ministry of Fisheries, 2019–2026) · CM4371")
st.subheader("Rajapakshe R.A.R.S. - 224223N")

tabs = st.tabs(["Problem & data", "Explore prices", "Model comparison",
                "Why these results?", "Price checker", "AI disclosure"])

with tabs[0]:
    c1, c2 = st.columns([3, 2])
    c1.markdown("""
### The question
Fish moves from the **Peliyagoda wholesale market** to retail markets around Colombo.
**Given this week's wholesale price of a species, what retail price should a shopper expect?**

### Approach
- **Target:** log markup, `log(retail / wholesale)`
- **Features:** wholesale price (now, last week, 4-week average), weekly change, market price index, season, species
- **Split by time:** train 2019–2023, test 2024 – Apr 2026
- **Models:** bagging, boosting, stacking and an RF + ANN hybrid, against two baselines
""")
    w = T["wide"]
    best = T["metrics_test"].iloc[0]
    c2.metric("Weekly reports", w["date"].nunique())
    c2.metric("Species", w["species"].nunique())
    c2.metric("Period", f"{w['date'].min():%b %Y} – {w['date'].max():%b %Y}")
    c2.metric(f"Best test MAPE ({best['model']})", f"{best['MAPE']:.2f}%")

with tabs[1]:
    w = T["wide"]
    species_list = sorted(w["species"].unique())
    sp = st.selectbox("Species", species_list, index=species_list.index("Skipjack tuna"))
    s = w[w["species"] == sp].melt(id_vars=["species", "date"], value_vars=["wholesale", "retail"],
                                   var_name="market", value_name="Rs/kg").dropna()
    fig = px.line(s, x="date", y="Rs/kg", color="market", markers=True, title=f"{sp}: weekly prices")
    fig.add_vline(x="2024-01-01", line_dash="dash", annotation_text="test period →")
    st.plotly_chart(fig, width="stretch")
    markup = (w["retail"] / w["wholesale"]).dropna().clip(0.5, 3)
    st.plotly_chart(px.histogram(markup, nbins=60, title=f"Retail ÷ wholesale (median {markup.median():.2f}×)"),
                    width="stretch")

with tabs[2]:
    m = T["metrics_test"].sort_values("MAPE", ascending=False)
    fig = px.bar(m, x="MAPE", y="model", color="family", orientation="h", color_discrete_map=COLORS,
                 text=m["MAPE"].round(2), labels={"MAPE": "Test MAPE % (lower is better)", "model": ""})
    st.plotly_chart(fig, width="stretch")
    cols = ["model", "family", "MAE", "RMSE", "MAPE", "R2", "train_MAPE", "cv_mape_mean", "cv_mape_std"]
    st.dataframe(T["metrics_test"][cols].round(2), hide_index=True, width="stretch")

    p = T["preds"]
    sp2 = st.selectbox("Species ", species_list, key="sp2", index=species_list.index("Skipjack tuna"))
    model_cols = [c for c in p.columns if c not in ["date", "species", "wholesale", "actual"]]
    picked = st.multiselect("Models", model_cols, default=[T["metrics_test"].iloc[0]["model"], "AdaBoost"])
    q = p[p["species"] == sp2]
    fig = go.Figure(go.Scatter(x=q["date"], y=q["actual"], name="actual", line=dict(color="green", width=3)))
    for name in picked:
        fig.add_trace(go.Scatter(x=q["date"], y=q[name], name=name))
    fig.update_layout(title="Predicted vs actual retail price", yaxis_title="Rs/kg")
    st.plotly_chart(fig, width="stretch")

with tabs[3]:
    st.subheader("Bias vs variance: train vs test error")
    bv = T["bias_variance"].sort_values("test_MAPE")
    fig = go.Figure([go.Bar(x=bv["model"], y=bv["train_MAPE"], name="train"),
                     go.Bar(x=bv["model"], y=bv["test_MAPE"], name="test")])
    st.plotly_chart(fig, width="stretch")

    st.subheader("Number of trees: bagging levels off, boosting keeps learning")
    c = T["ntrees_curve"].melt(id_vars=["model", "n_trees"], var_name="set", value_name="MAPE")
    st.plotly_chart(px.line(c, x="n_trees", y="MAPE", color="set", facet_col="model", markers=True, log_x=True),
                    width="stretch")

    c1, c2 = st.columns(2)
    c1.subheader("Correlation of model errors")
    c1.plotly_chart(px.imshow(T["corr"].round(2), text_auto=True, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1), width="stretch")
    c2.subheader("Stacking meta-model weights")
    c2.plotly_chart(px.bar(T["stack_weights"], x="meta_weight", y="base_model", orientation="h"),
                    width="stretch")

    c1, c2 = st.columns(2)
    imp = T["importance"].rename(columns={T["importance"].columns[0]: "feature"})
    c1.subheader("Feature importance (random forest)")
    c1.plotly_chart(px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h"),
                    width="stretch")
    c2.subheader("Raw price target vs markup target")
    c2.plotly_chart(px.bar(T["level_vs_markup"], x="model", y="MAPE", color="target", barmode="group"),
                    width="stretch")

    st.subheader("Error by species")
    se = T["species_error"].rename(columns={T["species_error"].columns[0]: "species"})
    st.plotly_chart(px.bar(se.melt(id_vars="species", var_name="model", value_name="MAPE %"),
                           x="species", y="MAPE %", color="model", barmode="group"), width="stretch")

    st.subheader("Findings")
    show_file("interpretation.md")


def build_row(species, date, wholesale, lag1, roll4, market_index):
    row = dict.fromkeys(B["features"], 0.0)
    row.update(log_wholesale=np.log(wholesale), log_wholesale_lag1=np.log(lag1),
               log_wholesale_roll4=np.log(roll4), wholesale_change=np.log(wholesale) - np.log(lag1),
               log_market_index=np.log(market_index), month=date.month, week=int(date.isocalendar().week),
               sw_monsoon=int(date.month in [5, 6, 7, 8, 9]), ne_monsoon=int(date.month in [12, 1, 2]))
    sp_col = "sp_" + re.sub(r"\W+", "_", species)
    if sp_col in row:
        row[sp_col] = 1.0
    return pd.DataFrame([row])[B["features"]]


with tabs[4]:
    st.subheader("Is this retail price fair?")
    c1, c2, c3 = st.columns(3)
    species = c1.selectbox("Species", B["species"], index=B["species"].index("Skipjack tuna"))
    date = pd.Timestamp(c1.date_input("Week", value=pd.Timestamp("2026-04-15")))
    default = float(round(B["latest_wholesale"].get(species, 1000.0)))
    wholesale = c2.number_input("Wholesale price this week (Rs/kg)", 50.0, 10000.0, default, step=10.0)
    lag1 = c2.number_input("Wholesale price last week (Rs/kg)", 50.0, 10000.0, default, step=10.0)
    roll4 = c3.number_input("Average wholesale, last 4 weeks (Rs/kg)", 50.0, 10000.0, default, step=10.0)
    mindex = c3.number_input("Market price index (Rs/kg)", 50.0, 10000.0,
                             float(round(B["latest_market_index"])), step=10.0)
    quoted = st.number_input("Retail price you were quoted (Rs/kg, optional)", 0.0, 20000.0, 0.0, step=10.0)

    X = build_row(species, date, wholesale, lag1, roll4, mindex)
    preds = {name: float(wholesale * np.exp(model.predict(X)[0])) for name, model in B["models"].items()}
    app_mape = T["app_models_test"].set_index("model")["MAPE"]
    best_name = B["best_model"] if B["best_model"] in preds else app_mape.idxmin()
    best_val = preds[best_name]
    error = app_mape[best_name] / 100
    lo, hi = best_val * (1 - 1.5 * error), best_val * (1 + 1.5 * error)

    st.metric(f"Expected retail price ({best_name})", f"Rs {best_val:,.0f} / kg")
    st.write(f"Typical range: **Rs {lo:,.0f} – {hi:,.0f}**")
    if quoted > hi:
        st.error("Above the expected range: the markup looks high.")
    elif 0 < quoted < lo:
        st.success("Below the expected range: a good deal.")
    elif quoted > 0:
        st.info("Within the expected range.")
    res = pd.DataFrame({"model": list(preds), "Rs/kg": list(preds.values())})
    st.plotly_chart(px.bar(res, x="Rs/kg", y="model", orientation="h", text=res["Rs/kg"].round(0)),
                    width="stretch")

with tabs[5]:
    show_file("AI_DISCLOSURE.md")