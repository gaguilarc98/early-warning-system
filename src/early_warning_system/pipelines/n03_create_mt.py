from .utils import *


########____CREATE MASTER TABLE____########


def bound_time_grid(ds, params):
    """Slice dataset with temporal bounds"""
    start_date = params['start_date']
    end_date = params['end_date']
    lon, lat, time = get_coordinates(ds)
    ds = ds.sel(time=slice(start_date, end_date))
    return ds


def create_raw_mt(
    params_mt: dict,
    params_t: dict,
    ds_po: xr.Dataset,
    *dataset_features
) -> xr.Dataset:
    """
    Align all feature datasets to ds_po base grid, lazy -- no data loaded.
    Args:
        - params_mt         : (dict) Parameters for master table, must contain 'sel_step'
        - params_t          : (dict) Temporal parameters, must contain 'start_date', 'end_date', 'freq'
        - ds_po             : (xr.Dataset) Base grid defining target lon/lat
        - *dataset_features : (list of xr.Dataset) Feature datasets at any grid resolution
    Returns:
        xr.Dataset lazy dataset on the ds_po grid with all feature variables
    """
    sel_step = params_mt['sel_step']
    date_range = get_date_array(params_t)

    aligned = []
    for ds in dataset_features:
        if 'step' in ds.dims:
            ds = ds.sel(step=sel_step).mean(dim='step')

        lon_name, lat_name, time_name = get_coordinates(ds)

        ds = ds.sel(
            {lon_name: ds_po['lon'].values, lat_name: ds_po['lat'].values},
            method='nearest'
        ).assign_coords(
            {lon_name: ds_po['lon'].values, lat_name: ds_po['lat'].values}
        )

        if time_name is not None:
            ds_times = pd.DatetimeIndex(ds[time_name].values)
            valid_times = date_range[date_range.isin(ds_times)]
            ds = ds.sel({time_name: valid_times})
        else:
            ds = ds.expand_dims({time_name or 'time': date_range})

        aligned.append(ds)

    ds_grid = xr.merge(aligned)
    ds_grid = add_time_coordinate(ds_grid, time_dim='time', level='month')
    ds_grid = add_time_coordinate(ds_grid, time_dim='time', level='week')
    ds_grid = add_time_coordinate(ds_grid, time_dim='time', level='yearmon')
    return ds_grid


def _extract_at_stations(
    ds_grid: xr.Dataset,
    df_target: pd.DataFrame,
    grid_coords: tuple = ('lon', 'lat'),
) -> pd.DataFrame:
    """
    Select nearest grid point per station, load only those pixels, return flat DataFrame.
    Args:
        - ds_grid     : (xr.Dataset) Lazy feature grid from create_raw_mt
        - df_target   : (pd.DataFrame) Station locations with lon/lat columns
        - grid_coords : (tuple) Coordinate names in ds_grid
    Returns:
        pd.DataFrame with columns (lon, lat, time, *feature_vars, month, week, yearmon)
    """
    lon_grid, lat_grid = grid_coords

    lons = ds_grid[lon_grid].values
    lats = ds_grid[lat_grid].values
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    df_coords = pd.DataFrame({
        lon_grid: lon_mesh.ravel(),
        lat_grid: lat_mesh.ravel(),
    })

    df_match = match_grid_points(df_coords, df_target, sample_coords=grid_coords, target_coords=grid_coords)

    ds_slice = select_coordinates(ds_grid, df_match, grid_coords)
    ds_slice = ds_slice.reset_index('points')

    df = ds_slice.to_dataframe().reset_index().drop(columns=['points'], errors='ignore')

    coord_map = df_match[
        [f'{lon_grid}_target', f'{lat_grid}_target', lon_grid, lat_grid]
    ].drop_duplicates()

    df = df.merge(
        coord_map,
        on=[lon_grid, lat_grid],
        how='left',
    ).drop(columns=[lon_grid, lat_grid]).rename(columns={
        f'{lon_grid}_target': lon_grid,
        f'{lat_grid}_target': lat_grid,
    })

    return df


def _add_target(
    df_mt: pd.DataFrame,
    df_target: pd.DataFrame,
    params_mt: dict,
) -> pd.DataFrame:
    """
    Merge target variable into master table and drop rows without observed target.
    Args:
        - df_mt     : (pd.DataFrame) Output of _extract_at_stations
        - df_target : (pd.DataFrame) Station observations including target variable
        - params_mt : (dict) Parameters for master table, must contain 'vars_y' and 'target'
    Returns:
        pd.DataFrame with target column merged in, rows without target dropped
    """
    vars_y = params_mt['vars_y']
    target = params_mt['target']
    merge_on = [c for c in ['lon', 'lat', 'time'] if c in df_target.columns]
    df_mt = df_mt.merge(df_target[vars_y], how='left', on=merge_on)
    return df_mt.dropna(subset=[target]).reset_index(drop=True)


########____ENDPOINTS____########


def create_mt_w_target(
    params_mt: dict,
    params_t: dict,
    ds_po: xr.Dataset,
    df_target: pd.DataFrame,
    *dataset_features
) -> pd.DataFrame:
    """
    Create Master Table with target variable for model training.
    Args:
        - params_mt         : (dict) Parameters for master table, must contain 'sel_step', 'vars_y', 'target'
        - params_t          : (dict) Temporal parameters, must contain 'start_date', 'end_date', 'freq'
        - ds_po             : (xr.Dataset) Base grid defining target lon/lat
        - df_target         : (pd.DataFrame) Station observations including target variable
        - *dataset_features : (list of xr.Dataset) Feature datasets at any grid resolution
    Returns:
        pd.DataFrame with feature and target columns at station locations, rows without target dropped
    """
    peril = params_mt['peril']
    params_mt_peril = params_mt[peril]
    ds_grid = create_raw_mt(params_mt_peril, params_t, ds_po, *dataset_features)
    df_mt = _extract_at_stations(ds_grid, df_target)
    return _add_target(df_mt, df_target, params_mt_peril)


def create_mt_predict(
    params_mt: dict,
    params_t: dict,
    ds_po: xr.Dataset,
    *dataset_features
) -> pd.DataFrame:
    """
    Create Master Table for inference on the full grid without target variable.
    Args:
        - params_mt         : (dict) Parameters for master table, must contain 'sel_step'
        - params_t          : (dict) Temporal parameters, must contain 'start_date', 'end_date', 'freq'
        - ds_po             : (xr.Dataset) Base grid defining target lon/lat
        - *dataset_features : (list of xr.Dataset) Feature datasets at any grid resolution
    Returns:
        pd.DataFrame with feature columns for every grid point
    """
    peril = params_mt['peril']
    params_mt_peril = params_mt[peril]
    ds_grid = create_raw_mt(params_mt_peril, params_t, ds_po, *dataset_features)
    return ds_grid.to_dataframe().reset_index()


def clean_mt(
    params_mt: dict,
    df_mt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert specified columns to category dtype.
    Args:
        - params_mt : (dict) Parameters for master table, must contain 'categorical_cols'
        - df_mt     : (pd.DataFrame) Output of create_mt_w_target or create_mt_predict
    Returns:
        pd.DataFrame with categorical columns cast to category dtype
    """
    categorical_cols = params_mt['categorical_cols']
    df_mt = df_mt.copy()
    for col in categorical_cols:
        if col in df_mt.columns:
            df_mt[col] = df_mt[col].astype('category')
        else:
            print(f'Warning: column {col} not found in DataFrame.')
    return df_mt