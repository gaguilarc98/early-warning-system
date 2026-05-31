"""
data_utils.py
-------------
Mock data generators and grid utilities for the early warning dashboard.

The mock functions mirror the exact schema returned by the Kedro pipeline so
the rest of the app is fully agnostic to the data source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon


# ── Mock alert thresholds (mirrors params_predict structure) ──────────────────

MOCK_PARAMS_PREDICT = {
    "alert_thresholds": {
        "coldspell": {"Red": 0.60, "Orange": 0.12, "None": 0.0},
        "rainfall":  {"Red": 0.40, "Orange": 0.10, "None": 0.0},
    }
}


# ── Threshold helper ──────────────────────────────────────────────────────────

def assign_alert_level(
    prob: float | pd.Series,
    thresholds: dict,
) -> str | pd.Series:
    """
    Map probability value(s) to alert level using a threshold dict of the form:
        {"Red": 0.60, "Orange": 0.12, "None": 0.0}
    """
    red    = thresholds["Red"]
    orange = thresholds["Orange"]

    def _scalar(p: float) -> str:
        if p >= red:
            return "Red"
        elif p >= orange:
            return "Orange"
        return "None"

    if isinstance(prob, pd.Series):
        return prob.apply(_scalar)
    return _scalar(prob)


# ── Mock AOI GeoDataFrame ─────────────────────────────────────────────────────

def generate_mock_aoi() -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame with polygon boundaries to overlay on the maps.
    Mirrors gdf_aoi from the Kedro catalog.

    Replace with:
        gdf_aoi = catalog.load("gdf_aoi")
    """
    rng = np.random.default_rng(7)

    dept_bounds = {
        "La Paz":     (-69.6, -17.5, -67.0, -9.7),
        "Oruro":      (-68.5, -19.5, -66.0, -17.0),
        "Potosí":     (-67.5, -22.9, -64.5, -19.0),
        "Cochabamba": (-66.5, -18.5, -64.0, -16.0),
        "Santa Cruz": (-63.5, -20.0, -57.5, -14.0),
        "Beni":       (-67.0, -15.5, -60.5, -10.0),
        "Pando":      (-69.6, -13.0, -65.5,  -9.7),
        "Tarija":     (-65.5, -22.9, -62.0, -20.5),
        "Chuquisaca": (-65.0, -21.5, -62.5, -18.5),
    }

    rows = []
    for dept, (lon0, lat0, lon1, lat1) in dept_bounds.items():
        n_prov = rng.integers(2, 5)
        lon_splits = np.linspace(lon0, lon1, n_prov + 1)
        for pi in range(n_prov):
            prov = f"{dept[:3].upper()}-P{pi+1}"
            n_mun = rng.integers(2, 5)
            lat_splits = np.linspace(lat0, lat1, n_mun + 1)
            for mi in range(n_mun):
                rows.append({
                    "idPolygon": f"{dept[:3].upper()}{pi}{mi}",
                    "dept":      dept,
                    "prov":      prov,
                    "mun":       f"{prov}-M{mi+1}",
                    "geometry":  Polygon([
                        (lon_splits[pi],     lat_splits[mi]),
                        (lon_splits[pi + 1], lat_splits[mi]),
                        (lon_splits[pi + 1], lat_splits[mi + 1]),
                        (lon_splits[pi],     lat_splits[mi + 1]),
                    ]),
                })

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ── Mock forecast DataFrames ──────────────────────────────────────────────────

def _make_pred_df(
    date: str,
    seed: int,
    thresholds: dict,
    spatial_phase: float = 0.0,
) -> pd.DataFrame:
    """Shared grid builder for both perils."""
    rng = np.random.default_rng(seed)
    lons = np.linspace(-69.6, -57.5, 80)
    lats = np.linspace(-22.9, -9.7,  80)
    lon_g, lat_g = np.meshgrid(lons, lats)

    prob = (
        0.30
        + 0.30 * np.sin(np.radians(lat_g) * 7 + spatial_phase)
        + 0.20 * np.cos(np.radians(lon_g) * 5)
        + 0.12 * rng.standard_normal(lon_g.shape)
    ).clip(0, 1)

    df = pd.DataFrame({
        "time":        pd.Timestamp(date),
        "lat":         lat_g.ravel(),
        "lon":         lon_g.ravel(),
        "pred_target": prob.ravel(),
    })
    df["alert_level"] = assign_alert_level(df["pred_target"], thresholds)
    return df


def _make_alerts_df(
    pred_df: pd.DataFrame,
    aoi_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Filter to Orange/Red rows and append hierarchy columns via spatial join.
    Mirrors df_update_alerts from the Kedro catalog.
    """
    filtered = pred_df[pred_df["alert_level"].isin(["Orange", "Red"])].copy()
    if filtered.empty:
        return pd.DataFrame(columns=[
            "idPolygon", "dept", "prov", "mun",
            "lon", "lat", "time", "pred_target", "alert_level",
        ])

    pts = gpd.GeoDataFrame(
        filtered,
        geometry=gpd.points_from_xy(filtered["lon"], filtered["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        pts,
        aoi_gdf[["idPolygon", "dept", "prov", "mun", "geometry"]],
        how="left",
        predicate="within",
    )
    out_cols = ["idPolygon", "dept", "prov", "mun",
                "lon", "lat", "time", "pred_target", "alert_level"]
    return joined[out_cols].reset_index(drop=True)


def generate_mock_data(date: str = "2025-01-01"):
    """
    Generate all five objects returned by load_data().

    Returns
    -------
    df_preds_coldspell  : pd.DataFrame  — full grid, coldspell
    df_alerts_coldspell : pd.DataFrame  — Orange/Red rows + hierarchy, coldspell
    df_preds_rainfall   : pd.DataFrame  — full grid, rainfall
    df_alerts_rainfall  : pd.DataFrame  — Orange/Red rows + hierarchy, rainfall
    gdf_aoi             : gpd.GeoDataFrame
    params_predict      : dict
    """
    params_predict = MOCK_PARAMS_PREDICT
    thresholds_cs  = params_predict["alert_thresholds"]["coldspell"]
    thresholds_rf  = params_predict["alert_thresholds"]["rainfall"]

    gdf_aoi = generate_mock_aoi()

    df_preds_coldspell  = _make_pred_df(date, seed=42, thresholds=thresholds_cs, spatial_phase=0.0)
    df_preds_rainfall   = _make_pred_df(date, seed=99, thresholds=thresholds_rf, spatial_phase=1.5)

    df_alerts_coldspell = _make_alerts_df(df_preds_coldspell, gdf_aoi)
    df_alerts_rainfall  = _make_alerts_df(df_preds_rainfall,  gdf_aoi)

    return (
        df_preds_coldspell,
        df_alerts_coldspell,
        df_preds_rainfall,
        df_alerts_rainfall,
        gdf_aoi,
        params_predict,
    )


# ── Grid reshape helper ───────────────────────────────────────────────────────

def forecast_to_grid(
    forecast_df: pd.DataFrame,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pivot a flat lon/lat DataFrame column into a 2-D grid for pcolormesh.

    Returns
    -------
    lon_grid, lat_grid, values  — all 2-D arrays
    """
    lons = np.sort(forecast_df["lon"].unique())
    lats = np.sort(forecast_df["lat"].unique())

    pivot = (
        forecast_df
        .pivot(index="lat", columns="lon", values=value_col)
        .reindex(index=lats, columns=lons)
    )

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lon_grid, lat_grid, pivot.values