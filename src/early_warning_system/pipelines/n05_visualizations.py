from .utils import *
from .a04_score_helpers import *

def plot_metrics_classif_by_group(df_preds, params):
    """
    Plot classification metrics grouped by subset and an additional groupby column.

    Parameters
    ----------
    df_preds : pd.DataFrame
        DataFrame with predictions for the full training dataset.
    params : dict
        Required keys:
            - 'groupby_cols'  : list[str], e.g. ["subset", "decadeCode"]
            - 'subset_col'    : str, the column identifying train/backtest/oos splits
            - 'target_col'    : str, ground-truth binary column
            - 'prob_col'      : str, predicted probability column (class 1)
        Optional keys:
            - 'palette'       : dict mapping subset values to colors
            - 'title'         : str, plot title
            - 'figsize'       : tuple, default (4.5, 7)
    
    Returns
    -------
    fig : matplotlib Figure
    df_metrics : pd.DataFrame (melted)
    """
    groupby_cols = params['groupby_cols']
    subset_col   = params['subset_col']
    peril        = params['peril']
    target_col   = params['target_col'][peril]
    prob_col     = params['prob_col']

    palette = params.get('palette', {
        "train":          "darkgreen",
        "backtest":       "darkred",
        "out_of_sample":  "darkblue",
    })
    title   = params.get('title', 'Classification metrics by subset')
    figsize = params.get('figsize', (4.5, 7))

    # Compute metrics per group
    df_metrics = (
        df_preds
        .groupby(groupby_cols)
        .apply(lambda g: get_metrics_classif(g[target_col], g[prob_col], name_set='value'))
        .reset_index(names= [*groupby_cols, 'metric'])
    )

    VALID_METRICS = ['AUC', 'GINI', 'KS', 'PR']

    df_metrics = df_metrics[df_metrics['metric'].isin(VALID_METRICS)].copy()

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df_metrics,
        y='metric',
        x='value',
        hue=subset_col,
        errorbar='sd',
        palette=palette,
        ax=ax,
    )
    ax.set_title(f'Metrics to evaluate the {peril} model')
    ax.legend(loc='lower right')
    ax.set_xlabel('')
    ax.set_ylabel('')

    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(1.5)

    plt.tight_layout()
    return fig



METRIC_FN_MAP = {
    'AUC':  lambda y, p: roc_auc_score(y, p) * 100,
    'GINI': lambda y, p: gini_score(y, p) * 100,
    'KS':   lambda y, p: ks_2samp(p[y == 0], p[y == 1]).statistic * 100,
    'F1':   lambda y, p: f1_score(y, np.where(p > 0.5, 1, 0)) * 100,
    'REC':  lambda y, p: recall_score(y, np.where(p > 0.5, 1, 0)) * 100,
    'PR':   lambda y, p: precision_at_target_recall(y, p, 0.25) * 100,
}


def plot_metric_scatter_geo(df_preds, gdf_aoi, params, params_s={}):
    """
    Scatter plot of a classification metric computed per station, overlaid on an AOI boundary.
    Args:
        df_preds : pd.DataFrame Full predictions DataFrame.
        gdf_aoi : gpd.GeoDataFrame AOI GeoDataFrame for boundary context.
        params : dict

            Required keys:
            - 'groupby_cols'  : list[str], e.g. ['decadeCode', 'lon_station', 'lat_station']
            - 'lon_col'       : str, longitude column name
            - 'lat_col'       : str, latitude column name
            - 'target_col'    : str, ground-truth column
            - 'prob_col'      : str, predicted probability column
            - 'metric'        : str, one of 'AUC', 'GINI', 'KS', 'F1', 'REC', 'PR'
            
            Optional keys:
            - 'aoi_filter'    : dict with keys 'col' and 'values' to filter gdf_aoi,
                                e.g. {'col': 'desc_cluster_topo', 'values': ['Valles', 'Altiplano']}
            - 'cmap'          : str or Colormap (default 'RdYlBu_r')
            - 'vmin'          : float (default 0)
            - 'vmax'          : float (default 100)
            - 'cbar_label'    : str
            - 'title'         : str
            - 'figsize'       : tuple (default (12, 9))
            - 'crs'           : str (default 'EPSG:4326')

    Returns
        fig : matplotlib Figure
        gdf_result : gpd.GeoDataFrame with 'metric' column per station
    """
    groupby_cols = params['groupby_cols']
    lon_col      = params['lon_col']
    lat_col      = params['lat_col']
    peril        = params['peril']
    target_col   = params['target_col'][peril]
    prob_col     = params['prob_col']
    metric_key   = params['metric'].upper()

    if metric_key not in METRIC_FN_MAP:
        raise ValueError(f"metric '{metric_key}' not recognised. Choose from: {list(METRIC_FN_MAP)}")
    metric_fn = METRIC_FN_MAP[metric_key]

    cmap       = params.get('cmap', plt.cm.RdYlBu_r)
    vmin       = params.get('vmin', 0)
    vmax       = params.get('vmax', 100)
    cbar_label = params.get('cbar_label', f'{metric_key} [0 - 100 %]')
    title      = params.get('title', f'{metric_key} by station')
    figsize    = params.get('figsize', (12, 9))
    crs        = params.get('crs', 'EPSG:4326')

    df_result = (
        df_preds
        .groupby(groupby_cols, as_index=False)
        .apply(lambda g: pd.Series({
            'metric': metric_fn(g[target_col], g[prob_col])
        }))
    )

    gdf_result = gpd.GeoDataFrame(
        df_result,
        geometry=gpd.points_from_xy(df_result[lon_col], df_result[lat_col]),
        crs=crs
    )

    # Subset geometry
    gdf_boundary = subset_geometry(gdf_aoi, params_s)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    gdf_boundary.boundary.plot(color='gray', ax=ax, linewidth=0.25)
    ax.scatter(
        gdf_result[lon_col], gdf_result[lat_col],
        c=gdf_result['metric'],
        norm=norm,
        cmap=cmap
    )
    ax.set_title(f'{metric_key} score by station for {peril}')

    ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
    ax.yaxis.set_major_formatter(FuncFormatter(format_latitude))
    ax.tick_params(axis='x', labelcolor='gray', labelsize=7, rotation=30)
    ax.tick_params(axis='y', labelcolor='gray', labelsize=7)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=cbar_label, fraction=0.03, pad=0.035)

    plt.tight_layout()
    return fig


def plot_precision_recall_vs_threshold(df_preds, params):
    """
    Compute and plot precision and recall across thresholds.

    Parameters
    ----------
    df_preds : pd.DataFrame
        Full predictions DataFrame.
    params : dict
        Required keys:
            - 'target_col'     : str, ground-truth binary column
            - 'prob_col'       : str, predicted probability column
        Optional keys:
            - 'n_thresholds'   : int (default 100)
            - 'subset_col'     : str, if provided plots one line per subset value
            - 'title'          : str
            - 'figsize'        : tuple (default (5, 5))
            - 'recall_color'   : str (default 'darkgreen')
            - 'precision_color': str (default 'darkred')

    Returns
    -------
    fig : matplotlib Figure
    df_pr : pd.DataFrame with threshold metrics (includes 'subset' col if subset_col given)
    """
    peril            = params['peril']
    target_col       = params['target_col'][peril]
    prob_col         = params['prob_col']
    n_thresholds     = params.get('n_thresholds', 100)
    subset_col       = params.get('subset_col', None)
    title            = params.get('title', 'Precision and Recall vs Threshold')
    figsize          = params.get('figsize', (5, 5))
    recall_color     = params.get('recall_color', 'darkgreen')
    precision_color  = params.get('precision_color', 'darkred')

    def _compute(df):
        thresholds = np.linspace(0, 1, n_thresholds)
        records = []
        y_true = df[target_col].astype(int).values
        y_prob = df[prob_col].values

        for t in thresholds:
            y_pred = (y_prob > t).astype(int)

            TP = np.sum((y_true == 1) & (y_pred == 1))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            TN = np.sum((y_true == 0) & (y_pred == 0))
            FN = np.sum((y_true == 1) & (y_pred == 0))

            records.append({
                'threshold'    : t,
                'fraction_days': y_pred.sum() / len(df),
                'precision'    : TP / (TP + FP) if (TP + FP) > 0 else 0,
                'recall'       : TP / (TP + FN) if (TP + FN) > 0 else 0,
                'fpr'          : FP / (FP + TN) if (FP + TN) > 0 else 0,
                'fnr'          : FN / (FN + TP) if (FN + TP) > 0 else 0,
                'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN),
            })
        return pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=figsize)

    if subset_col is not None:
        subsets = df_preds[subset_col].unique()
        # Use a cycling linestyle per subset to keep recall/precision color-coded
        linestyles = ['-', '--', ':', '-.']
        dfs = []
        for i, subset in enumerate(subsets):
            ls = linestyles[i % len(linestyles)]
            df_sub = _compute(df_preds[df_preds[subset_col] == subset])
            df_sub[subset_col] = subset
            dfs.append(df_sub)
            ax.plot(df_sub['threshold'], df_sub['recall'],
                    label=f'Recall ({subset})', color=recall_color, linestyle=ls)
            ax.plot(df_sub['threshold'], df_sub['precision'],
                    label=f'Precision ({subset})', color=precision_color, linestyle=ls)
        df_pr = pd.concat(dfs, ignore_index=True)
    else:
        df_pr = _compute(df_preds)
        ax.plot(df_pr['threshold'], df_pr['recall'],
                label='Recall', color=recall_color)
        ax.plot(df_pr['threshold'], df_pr['precision'],
                label='Precision', color=precision_color, linestyle='--')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric')
    ax.set_title(f'Precision and Recall vs Threshold for {peril}')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return fig