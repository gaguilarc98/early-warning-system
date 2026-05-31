"""
app.py
------
Early Warning System Dashboard — Streamlit entry point.

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit matplotlib pandas geopandas shapely numpy
"""

from __future__ import annotations

from datetime import datetime
import pandas as pd
import geopandas as gpd
import yaml

import streamlit as st

# ── Kedro imports (active in production) ─────────────────────────────────────
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
bootstrap_project(Path.cwd())

from data_utils import generate_mock_data, forecast_to_grid
from plot_utils import plot_probability, plot_alert_level


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Early Warning System · Suyana",
    page_icon="⚠️",
    layout="wide",
)


# ── Load data ─────────────────────────────────────────────────────────────────
ROOT_PATH = Path(__file__).resolve().parents[3]

@st.cache_data(show_spinner="Running forecast pipeline…")

def load_data(date: str):
    #date_fct = datetime.now().strftime('%Y-%m-%d')
    df_preds_coldspell  = pd.read_parquet(ROOT_PATH / "data/outputs/ews_coldspell_update_predictions.parquet")
    df_alerts_coldspell = pd.read_parquet(ROOT_PATH / "data/outputs/ews_coldspell_update_alerts.parquet")
    df_preds_rainfall   = pd.read_parquet(ROOT_PATH / "data/outputs/ews_rainfall_update_predictions.parquet")
    df_alerts_rainfall  = pd.read_parquet(ROOT_PATH / "data/outputs/ews_rainfall_update_alerts.parquet")
    gdf_aoi             = gpd.read_parquet(ROOT_PATH / "data/geometries/location_geometries.parquet")

    with open(ROOT_PATH / "conf/base/parameters.yml") as f:
        params_predict = yaml.safe_load(f)["predict"]
    
    return (
        df_preds_coldspell, df_alerts_coldspell,
        df_preds_rainfall,  df_alerts_rainfall,
        gdf_aoi, params_predict,
    )

# Kedro data

#def load_data(date: str):
#    date_fct = datetime.now().strftime('%Y-%m-%d')
#    runtime_params_tmin = {
#        'country':    'bolivia',
#        'provider':   'ECMWF',
#        'field':      'tmin',
#        'peril':      'coldspell',
#        'start_date': date_fct,
#        'end_date':   date_fct,
#    }
#    with KedroSession.create(runtime_params=runtime_params_tmin) as session:
#        catalog = session.load_context().catalog
#        gdf_aoi              = catalog.load("gdf_aoi")
#        df_preds_coldspell   = catalog.load("df_update_preds")
#        df_alerts_coldspell  = catalog.load("df_update_alerts")
#        params_predict       = catalog.load("params:predict")
#
#    runtime_params_prcp = {
#        'country':    'bolivia',
#        'provider':   'ECMWF',
#        'field':      'prcp',
#        'peril':      'rainfall',
#        'start_date': date_fct,
#        'end_date':   date_fct,
#    }
#    with KedroSession.create(runtime_params=runtime_params_prcp) as session:
#        catalog = session.load_context().catalog
#        df_preds_rainfall    = catalog.load("df_update_preds")
#        df_alerts_rainfall   = catalog.load("df_update_alerts")
#
#    return (
#        df_preds_coldspell,
#        df_alerts_coldspell,
#        df_preds_rainfall,
#        df_alerts_rainfall,
#        gdf_aoi,
#        params_predict,
#    )


# ── Mock data (comment out when using Kedro) ──────────────────────────────────
# To switch to production, comment the next two lines and uncomment load_data above.

#@st.cache_data(show_spinner="Loading mock data…")
#def load_data(date: str):  # noqa: F811
#    return generate_mock_data(date=date)

# ─────────────────────────────────────────────────────────────────────────────

(
    df_preds_coldspell,
    df_alerts_coldspell,
    df_preds_rainfall,
    df_alerts_rainfall,
    gdf_aoi,
    params_predict,
) = load_data(datetime.now().strftime('%Y-%m-%d'))

thresholds = params_predict["alert_thresholds"]


# ── Peril config ──────────────────────────────────────────────────────────────

PERIL_CONFIG = {
    "Coldspell": {
        "icon":        "🌨️",
        "df_preds":    df_preds_coldspell,
        "df_alerts":   df_alerts_coldspell,
        "thresholds":  thresholds["coldspell"],
        "prob_label":  "P(coldspell)",
        "filename":    "alerts_coldspell.csv",
    },
    "Rainfall": {
        "icon":        "🌧️",
        "df_preds":    df_preds_rainfall,
        "df_alerts":   df_alerts_rainfall,
        "thresholds":  thresholds["rainfall"],
        "prob_label":  "P(extreme rainfall)",
        "filename":    "alerts_rainfall.csv",
    },
}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚠️ Early Warning System")
    st.caption("Suyana · Climate Insurance")
    st.markdown("---")

    peril = st.radio("Peril", list(PERIL_CONFIG.keys()), horizontal=False)
    cfg   = PERIL_CONFIG[peril]

    thresh = cfg["thresholds"]
    st.markdown("---")
    st.markdown(f"**{cfg['icon']} {peril} alert thresholds**")
    st.markdown(
        f"- 🔴 Red &nbsp;&nbsp;&nbsp; ≥ {thresh['Red']:.0%}\n"
        f"- 🟠 Orange ≥ {thresh['Orange']:.0%}\n"
        f"- ⬛ None &nbsp;&nbsp; < {thresh['Orange']:.0%}"
    )
    st.markdown("---")
    forecast_date = datetime.now().strftime('%Y-%m-%d')
    st.caption(f"Forecast date: **{forecast_date}**")


# ── Active datasets ───────────────────────────────────────────────────────────

df_preds  = cfg["df_preds"]
df_alerts = cfg["df_alerts"]

lon_grid, lat_grid, prob_grid  = forecast_to_grid(df_preds, "pred_target")
lon_grid, lat_grid, alert_grid = forecast_to_grid(df_preds, "alert_level")


# ── KPI strip ─────────────────────────────────────────────────────────────────

st.title(f"{cfg['icon']} {peril} Risk Dashboard")
st.caption(f"Forecast date: **{forecast_date}**")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Grid points",         f"{len(df_preds):,}")
k2.metric("Max probability",     f"{df_preds['pred_target'].max():.1%}")
k3.metric("Orange alert points", int((df_preds["alert_level"] == "Orange").sum()))
k4.metric("Red alert points",    int((df_preds["alert_level"] == "Red").sum()))

st.markdown("---")


# ── Maps ──────────────────────────────────────────────────────────────────────

col_prob, col_alert = st.columns([7,6], gap="medium")

with col_prob:
    st.subheader("Predicted probability")
    fig_prob = plot_probability(
        lon_grid, lat_grid, prob_grid,
        date=forecast_date,
        boundaries_gdf=gdf_aoi,
        prob_label=cfg["prob_label"],
        thresholds=thresh,
    )
    st.pyplot(fig_prob, use_container_width=True)

with col_alert:
    st.subheader("Alert level")
    fig_alert = plot_alert_level(
        lon_grid, lat_grid, alert_grid,
        date=forecast_date,
        boundaries_gdf=gdf_aoi,
        peril_label=peril,
    )
    st.pyplot(fig_alert, use_container_width=True)

st.markdown("---")


# ── Alert table + downloads ───────────────────────────────────────────────────

st.subheader(f"Alert regions — {peril} (Orange & Red)")

col_table, col_dl = st.columns([3, 1], gap="large")

with col_table:
    if df_alerts.empty:
        st.info(f"No Orange or Red alerts for {peril} on {forecast_date}.")
    else:
        LEVEL_ICON = {"Orange": "🟠", "Red": "🔴"}
        display = df_alerts.copy()
        display["alert_level"] = display["alert_level"].map(
            lambda x: f"{LEVEL_ICON.get(x, '')} {x}"
        )
        display["pred_target"] = display["pred_target"].map("{:.2%}".format)
        display["time"]        = display["time"].astype(str)
        st.dataframe(
            display.rename(columns={
                "idPolygon":   "ID",
                "dept":        "Department",
                "prov":        "Province",
                "mun":         "Municipality",
                "lon":         "Lon",
                "lat":         "Lat",
                "time":        "Date",
                "pred_target": "Probability",
                "alert_level": "Alert",
            }),
            use_container_width=True,
            hide_index=True,
        )

with col_dl:
    st.markdown("##### Downloads")

    for label, df, filename in [
        ("⬇️ Coldspell alerts", df_alerts_coldspell, "alerts_coldspell.csv"),
        ("⬇️ Rainfall alerts",  df_alerts_rainfall,  "alerts_rainfall.csv"),
    ]:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=label,
            data=csv_bytes,
            file_name=f"{filename.replace('.csv', '')}_{forecast_date}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=df.empty,
        )

    st.caption(
        f"Active: **{peril}** · {len(df_alerts)} alert points\n\n"
        "Columns: idPolygon, dept, prov, mun, lon, lat, time, pred_target, alert_level"
    )