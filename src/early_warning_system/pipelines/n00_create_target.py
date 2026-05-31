from .utils import *

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler


#——————————————————————————————————————————————
# CREATE POPULATION GRID
#——————————————————————————————————————————————


def create_grid(
    params_s: dict, 
) -> xr.Dataset:
    """
    Create target population grid
    Args:
        - params_s : (dict) Parameters containing spatial specifications to create the target population.
            Spatial: lon_min, lon_max, lat_min, lat_max, nodes_by_deg
        - params_t : (dict) Parameters containing temporal specifications to create the target population.
            Time: start_date, end_date, freq, date_format
    Returns:
        - xr.Dataset with the grid to work with
    """
    minx, maxx, maxy, miny = params_s['minx'], params_s['maxx'], params_s['maxy'], params_s['miny']
    nodes_by_deg = params_s.get('nodes_by_deg', 10)
    '''
    start_date, end_date, freq, fmt = params_t['start_date'], params_t['end_date'], params_t['freq'], params_t['date_format']
    '''

    # Calculate proper bounds to cover the area
    #minx = get_round_value(np.floor(minx*nodes_by_deg)/nodes_by_deg, nodes_by_deg)
    #maxx = get_round_value(np.ceil(maxx*nodes_by_deg)/nodes_by_deg, nodes_by_deg)
    #miny = get_round_value(np.floor(miny*nodes_by_deg)/nodes_by_deg, nodes_by_deg)
    #maxy = get_round_value(np.ceil(maxy*nodes_by_deg)/nodes_by_deg, nodes_by_deg)

    #lon_array = np.arange(minx, maxx + 1/nodes_by_deg, 1/nodes_by_deg)
    #lat_array = np.arange(maxy, miny - 1/nodes_by_deg, -1/nodes_by_deg)
    lon_array = np.arange(minx, maxx, 1/nodes_by_deg)
    lat_array = np.arange(maxy, miny, -1/nodes_by_deg)

    # Create xarray Dataset
    ds_po = xr.Dataset(coords={
        'lon': ('lon', lon_array),
        'lat': ('lat', lat_array),
        #'time': ('time', date_range)
    })

    return ds_po 
    

def create_geometry(ds_po: xr.Dataset) -> gpd.GeoDataFrame:
    """
    Create target population geometry object from dataset.
    Args:
        ds_po: Dataset containing the coordinates for the grid.
    Returns:
        GeoDataFrame with point geometries for each grid cell.
    """
    lon_grid, lat_grid = np.meshgrid(ds_po['lon'].values, ds_po['lat'].values)

    gdf_po = gpd.GeoDataFrame(
        {'lon': lon_grid.flatten(), 'lat': lat_grid.flatten()},
        geometry=gpd.points_from_xy(lon_grid.flatten(), lat_grid.flatten()),
        crs='EPSG:4326'
    )

    return gdf_po


#——————————————————————————————————————————————
# CREATE CLUSTER
#——————————————————————————————————————————————


def create_cluster(df_orig, variable, agg_func='mean', group_cols=['decadeCode']):
    df = df_orig.copy()
    
    # Create DataFrame with coordinates
    df_cluster = df[group_cols + ['latitude', 'longitude', 'altitude']].drop_duplicates()
    df_cluster = df_cluster.set_index(group_cols)
    
    # Create DataFrame with climatoology
    df_clim = df.pivot_table(
        index=group_cols, 
        columns=['week'], 
        values=[variable], 
        aggfunc=agg_func
    )
    df_clim.columns = [col[0] + '_' + str(col[1]) for col in df_clim.columns]
    df_clim = df_clim.drop(columns=[f'{variable}_53'])

    df_raw = pd.merge(df_cluster, df_clim, left_index=True, right_index=True, how="inner")
    
    # Scale variables
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_raw)

    # Get first 4 principal components
    pca = PCA(n_components=4, random_state=42)
    pca.fit(df_scaled)

    df_pca = pd.DataFrame(
        pca.transform(df_scaled), 
        index=df_raw.index, 
        columns=['comp_'+str(i) for i in range(4)]
    )

    km = KMeans(n_clusters=3, n_init=18, random_state=42)
    df_pca['cluster'] = km.fit_predict(df_pca)
    df_pca.reset_index(inplace=True)

    return df_pca
    
#——————————————————————————————————————————————
# CREATE COLDSPELLS TARGET
#——————————————————————————————————————————————


def clean_tn_station_data(
    df_station: pd.DataFrame
):
    var_replace = [
        'misTempInt', 'asyRoundPat', 'meaPrec', 'meaPrecInd', 'meaPrecInc', 
        'othQualProb', 'grossInhoFreq', 'monOutlierAcmAll', 'monOutlierAcmSel'
    ]
    for var in var_replace:
        df_station[var] = df_station[var].mask(df_station[var]<=-99, np.nan)

    df_station['flagNA'] = np.where(
        ((df_station['TNUP']==1) & (df_station['DTLO']==1)) |
        ((df_station['TNLO']==1) & (df_station['DTLO']==1)) |
        ((df_station['TNXL']==1)) |
        ((df_station['TNJU']==1)) | 
        ((df_station['UPTN']==1)) |
        ((df_station['LOTN']==1) & (df_station['TNLO']==1)), 1, 0
    )
    df_station['flagNA_2'] = np.where(
        ((df_station['othQualProb']>=2) & (df_station['misTempInt']>=2)) |
        ((df_station['othQualProb']>=3) & (df_station['grossInhoFreq']>=3)) | 
        ((df_station['monOutlierAcmAll']>=1) & (df_station['monOutlierAcmSel']>=1)),
        1, 0
    )

    df_station['TN_clean'] = np.where(
        (df_station['flagNA'] == 1) |  (df_station['flagNA_2'] == 1), 
        np.nan, df_station['TN']
    )
    return df_station


def process_tn_station_data(
    df_var: pd.DataFrame, 
    params_t: dict, 
    params_s: dict,
    group_cols: list = ['decadeCode']
):
    """
    Creates a GeoDataframe with the information about the target variable for a subset of the grid
    Args:
        - ds_po : (xr.Dataset) Dataset containing the target population grid
        - df_tn : (pd.DataFrame) Dataframe with the time series of target variable
        - df_tn_metadata : (pd.DataFrame) Dataframe with the coordinates and other info about the weather stations
        - gdf_cluster : (pd.DataFrame) Dataframe with the geometries selected for our AOI
        - params_t : (dict) Dictionary with temporal parameters for slicing in time
        - params_target : (dict) Dictionary with spatial parameters for slicing in space
    Returns:
        pd.DataFrame with the target variable for a slice of the population target
    """
    start_date, end_date, fmt = params_t['start_date'], params_t['end_date'], params_t['date_format']
    start_date = pd.to_datetime(start_date, format=fmt)
    end_date = pd.to_datetime(end_date, format=fmt)

    # Clean data for in-situ stations
    #df_var = clean_tn_station_data(df_var)
    df_var['TN_clean'] = df_var['TN']

    # Format data
    df_var = df_var.rename(columns={
        'Date': 'time',
    })
    df_var['time'] = pd.to_datetime(df_var['time'])

    # Create cluster
    df_var = add_time_coordinate_df(df_var, 'time', 'week')

    df_cluster = create_cluster(df_var, 'TN_clean', agg_func='mean', group_cols=group_cols)
    df_var = df_var.merge(
        df_cluster[group_cols + ['cluster']],
        how = 'left',
        on = group_cols
    )

    # Select relevant columns
    df_var = df_var[
        group_cols + [ 
        'longitude',
        'latitude',
        'time', 
        'cluster',
        'TN_clean'
    ]].copy()

    # Filter data within time frame and AOI
    df_var = df_var[
        (df_var['latitude'] >= params_s['miny']) & (df_var['latitude'] <= params_s['maxy']) &
        (df_var['longitude'] >= params_s['minx']) & (df_var['longitude'] <= params_s['maxx']) 
    ].copy()
    df_var = df_var[(df_var['time']>= start_date) & (df_var['time']<= end_date)].copy()

    return df_var


def get_window_days(doy, window=30):
    """Return list of days-of-year within ±window of doy (wrap around at 365)."""
    days = np.arange(doy - window, doy + window + 1)
    return ((days - 1) % 365) + 1


def get_percentile_climatology_df(
    df_orig: pd.DataFrame, 
    variable,
    group_cols: list = ['decadeCode'],
    clim_window = 7,
):
    """
    Creates a climatology of the selected variable grouping on the selected columns
    Args:
        -  df_orig: pd.DataFrame containing the original dataset with the variable to calculate climatology

    """
    # Check if dayof year is present as a coordinate:
    if 'dayofyear' not in df_orig.columns:
        raise ValueError('dayofyear is not present in the DataFrame to compute daily climatologies')
    
    df_target = df_orig.copy()
    df_list = []
    # For each day of year compute the climatology within a time window of days before and after
    for day in range(1, 366):
        list_days = get_window_days(day, window=clim_window)
        df_base = df_target[df_target['dayofyear'].isin(list_days)].copy()
        # For each percentile compute the climatology of the variable along the grouping columns
        df_base[f'perc_{variable}'] = df_base.groupby(group_cols)[variable].transform(
            lambda x: np.round(x.rank(method="average", pct=True)*100, 4)
        )
        df_base = df_base[
            df_base['dayofyear']==day
        ][
            group_cols + ['time', f'perc_{variable}']
        ].copy()
        df_list.append(df_base)

    df_full = pd.concat(df_list, axis=0, ignore_index=True)

    df_full = df_full.sort_values(by=group_cols+['time'])
    return df_full


def create_percentile_tn_target(
    df_target,
    params_target,
    suffix
):
    #prcp_thresholds = params_target['prcp_thresholds']
    perc_thresholds = params_target['perc_thresholds']
    temp_days = params_target['temp_days']
    variable = params_target['variable']
    group_cols = params_target['group_cols']

    df_merge = df_target[group_cols + ['time', variable]].copy()

    df_merge[f'{variable}_bin_{suffix}'] = np.where(
        (df_merge[variable] <= 0) & ~(df_merge[variable].isna()), 1, 0
    )

    # Add dayofyear coordinate to compute climatologies later
    df_merge = add_time_coordinate_df(df_merge, level='dayofyear')

    df_merge.sort_values(by= group_cols + ['time'], inplace=True)

    # Create target if the daily minimum temperature is lower than the p percentile for d consecutive days
    for days in temp_days:
        df_merge[f'tmin_last{days}d_{suffix}'] = (
            df_merge
            .groupby(group_cols)[variable]
            .rolling(window=days, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        target_var = f'tmin_next{days}d_{suffix}'
        df_merge[target_var] = (
            df_merge
            .groupby(group_cols)[f'tmin_last{days}d_{suffix}']
            .shift(-(days-1))
        )

        # Get percentiles of prcp in the next d days
        df_clim = get_percentile_climatology_df(
            df_merge,
            target_var,
            group_cols = group_cols,
            clim_window=3
        )
        df_merge = df_merge.merge(
            df_clim,
            how = 'left',
            on = group_cols + ['time']
        )

        for perc in perc_thresholds:
            df_merge[f'target_P{perc}_{days}d_{suffix}'] = np.where(
                df_merge[f'perc_{target_var}'] <= perc, 1, 0
            )
            
        df_merge.drop(columns=[f'tmin_last{days}d_{suffix}'], inplace=True)

    return df_merge


def get_rolling_climatology_df(
    df_orig: pd.DataFrame, 
    variable,
    group_cols: list = ['decadeCode'],
    clim_window = 7,
):
    """
    Creates a climatology of the selected variable grouping on the selected columns
    Args:
        -  df_orig: pd.DataFrame containing the original dataset with the variable to calculate climatology

    """
    # Check if dayof year is present as a coordinate:
    if 'dayofyear' not in df_orig.columns:
        raise ValueError('dayofyear is not present in the DataFrame to compute daily climatologies')
    
    df_target = df_orig.copy()
    df_list = []
    # For each day of year compute the climatology within a time window of days before and after
    for day in range(1, 366):
        list_days = get_window_days(day, window=clim_window)
        df_base = df_target[df_target['dayofyear'].isin(list_days)].copy()
        # For each percentile compute the climatology of the variable along the grouping columns
        df_base[f'clim_{variable}_mean'] = df_base.groupby(group_cols)[variable].transform(
            lambda x: np.round(np.mean(x), 4)
        )
        df_base[f'clim_{variable}_std'] = df_base.groupby(group_cols)[variable].transform(
            lambda x: np.round(np.std(x), 4)
        )
        df_base = df_base[
            df_base['dayofyear']==day
        ][
            group_cols + ['time', f'clim_{variable}_mean', f'clim_{variable}_std']
        ].copy()
        df_list.append(df_base)

    df_full = pd.concat(df_list, axis=0, ignore_index=True)

    df_full = df_full.sort_values(by=group_cols+['time'])
    return df_full


def create_tn_target(
    df_target,
    params_target,
    suffix
):
    #prcp_thresholds = params_target['prcp_thresholds']
    #perc_thresholds = params_target['perc_thresholds']
    sd_thresholds = params_target['sd_thresholds']
    temp_days = params_target['temp_days']
    variable = params_target['variable']
    group_cols = params_target['group_cols']

    df_merge = df_target[group_cols + ['time', variable]].copy()

    df_merge[f'{variable}_bin_{suffix}'] = np.where(
        (df_merge[variable] <= 0) & ~(df_merge[variable].isna()), 1, 0
    )

    # Add dayofyear coordinate to compute climatologies later
    df_merge = add_time_coordinate_df(df_merge, level='dayofyear')

    # Get climatology of daily temperature
    df_clim = get_rolling_climatology_df(
        df_merge,
        variable,
        group_cols = group_cols,
        clim_window=7
    )
    df_merge = df_merge.merge(
        df_clim,
        how = 'left',
        on = group_cols + ['time']
    )

    df_merge.sort_values(by= group_cols + ['time'], inplace=True)

    for t in sd_thresholds:
        df_merge[f'flag_tmin_{t}sd_{suffix}'] = np.where(
            df_merge[f'{variable}'] <= (df_merge[f'clim_{variable}_mean'] - t*df_merge[f'clim_{variable}_std']), 1, 0
        )
        # Create target if there are d consecutive cold days
        for days in temp_days:
            target_var = f'n_{t}sd_next{days}d_{suffix}'
            df_merge[target_var] = ( #f'n_{t}sd_next{days}d_{suffix}'
                df_merge
                .groupby(group_cols)[f'flag_tmin_{t}sd_{suffix}']
                .rolling(window=days, min_periods=days)
                .sum()
                .shift(-(days-1))
                .reset_index(level=0, drop=True)
            )
            #df_merge[target_var] = (
            #    df_merge
            #    .groupby(group_cols)[f'n_{t}sd_last{days}d_{suffix}']
            #    .shift(-(days-1))
            #)
            df_merge[f'target_{t}sd_{days}d_{suffix}'] = np.where(
                df_merge[target_var]>=days, 1, 0
            )
            #df_merge.drop(columns=[f'n_{t}sd_last{days}d_{suffix}'], inplace=True)

    return df_merge


def create_cold_spells_target( 
    df_tn: pd.DataFrame, 
    ds_era5: xr.Dataset,
    ds_po: xr.Dataset,
    gdf_cluster: gpd.GeoDataFrame, 
    params_t: dict, 
    params_target: dict,
    params_s: dict
):
    """
    Creates a GeoDataframe with the information about the target variable for a subset of the grid
    Args:
        - df_tn : (pd.DataFrame) Dataframe with the time series of target variable
        - ds_era5 : (pd.DataFrame) Dataframe with the time series of minimum temperature from ERA5-Land
        - ds_po : (xr.Dataset) Dataset containing the target population grid
        - gdf_cluster : (pd.DataFrame) Dataframe with the geometries selected for our AOI
        - params_t : (dict) Dictionary with temporal parameters for slicing in time
        - params_target : (dict) Dictionary with  parameters to build the target variables
        - params_s : (dict) Dictionary with spatial parameters for slicing in space
    Returns:
        pd.DataFrame with the target variable for a slice of the target population
    """
    start_date, end_date, freq, fmt = params_t['start_date'], params_t['end_date'], params_t['freq'], params_t['date_format']
    start_date = pd.to_datetime(start_date, format=fmt)
    end_date = pd.to_datetime(end_date, format=fmt)
    sd_thresholds = params_target['sd_thresholds']
    temp_days = params_target['temp_days']
    group_cols = params_target['group_cols']

    # Step 1: Create auxiliary dataframe with the coordinates and the grouping cols from weather stations
    df_aux_metadata = df_tn[group_cols + ['longitude', 'latitude']].drop_duplicates().copy()

    # Step 2: Compute the target variable options from weather station records
    # Clean weather station dataset
    df_clean = process_tn_station_data(df_tn, params_t, params_s, group_cols)
    df_cluster = df_clean[group_cols + ['cluster']].drop_duplicates().copy()

    # Create a full cross-join dataset for all dates and locations
    df_dates = pd.DataFrame({
        'time': pd.date_range(start_date, end_date, freq=freq)
    })
    df_grid = df_clean[group_cols].drop_duplicates()
    df_grid = df_grid.merge(df_dates, how='cross')
    # Merge clean dataset with cross-join to fill in any gaps
    df_clean = df_grid.merge(
        df_clean,
        how = 'left',
        on = group_cols + ['time']
    ).rename(columns={'TN_clean': 'tmin_lfa'})

    # Compute the target variables
    #params_target['variable'] = 'TN'
    #df_target_lfa = create_tn_target(df_clean, params_target, 'lfa')
    #df_target_lfa = df_cluster.merge(df_target_lfa, how='left', on=group_cols)
    #print('Target from weather stations succesfully created')

    # Step 3: Compute the target variable options from ERA5
    # Match station coordinates to CHIRPS coordinates
    df_coords = match_grid_points(
        ds_era5, 
        df_aux_metadata, 
        sample_coords=('lon', 'lat'), 
        target_coords=('longitude', 'latitude')
    )
    # Select coordinates from ERA5 closest to the stations
    ds_slice = select_coordinates(ds_era5, df_coords)
    ds_slice = ds_slice.reset_index('points')
    # Convert the slice of ERA5 to a dataframe
    df_era5 = ds_slice.to_dataframe().reset_index().drop(columns=['spatial_ref'])
    # Append metadata to dataframe to compute the target by grouping columns
    df_era5 = df_era5.merge(
        df_coords,
        how = 'left',
        on = ['lon', 'lat']
    ).rename(columns={'tmin': 'tmin_chirts'})
    df_era5 = df_era5.drop(columns=['points', 'lon', 'lat'])
    # Filter dataset within time frame
    df_era5 = df_era5[
        (df_era5['time']>= start_date) 
        & (df_era5['time']<= end_date)
    ].copy()

    # Compute the target variables
    #params_target['variable'] = 't2m'
    #df_target_era5 = create_tn_target(df_era5, params_target, 'era5')
    #print('Target from gridded data succesfully created')

    # Step 4: Merge in-situ target dataset with CHIRPS target dataset
    df_target = df_aux_metadata.merge(
        df_clean,
        how = 'left',
        on = group_cols + ['longitude', 'latitude']
    ).merge(
        df_era5,
        how = 'left',
        on = group_cols + ['longitude', 'latitude', 'time']
    )

    # Adjust bias in ERA5 dataset
    df_target['month'] = df_target['time'].dt.month
    df_target['tmin_chirts'] = df_target['tmin_chirts'] #- 273.15
    df_target['mean_tmin_lfa'] = df_target.groupby(group_cols + ['month'])['tmin_lfa'].transform('mean')
    df_target['mean_tmin_chirts'] = df_target.groupby(group_cols + ['month'])['tmin_chirts'].transform('mean')
    df_target['std_tmin_lfa'] = df_target.groupby(group_cols + ['month'])['tmin_lfa'].transform('std')
    df_target['std_tmin_chirts'] = df_target.groupby(group_cols + ['month'])['tmin_chirts'].transform('std')

    df_target['tmin_chirts_adj'] = (
        df_target['mean_tmin_lfa'] + (df_target['std_tmin_lfa']/df_target['std_tmin_chirts']) *
        (df_target['tmin_chirts'] - df_target['mean_tmin_chirts'])
    )

    df_target['tmin'] = np.apply_along_axis(np.nanmean, 1, df_target[['tmin_lfa', 'tmin_chirts_adj']])
    
    params_target['variable'] = 'tmin'
    #df_target_aux = create_percentile_tn_target(df_target, params_target, 'final')
    df_target = df_target.drop(columns=['month'])
    df_target_aux = create_tn_target(df_target, params_target, 'final')

    df_target = df_target.merge(
        df_target_aux.drop(columns=['tmin']),
        how = 'left',
        on = group_cols + ['time']
    )

    print('Target from gridded data succesfully created')    

    # Step 5: Add metadata (coordinates) from population grid
    df_coords_po = match_grid_points(
        ds_po, 
        df_aux_metadata, 
        sample_coords=('lon', 'lat'), 
        target_coords=('longitude', 'latitude')
    )
    df_target = df_coords_po.merge(
        df_target,
        how = 'left',
        on = group_cols + ['longitude', 'latitude']
    )

    # (Optional) Add metadata from geometry file
    gdf_aux_metadata = gpd.GeoDataFrame(
        df_aux_metadata,
        geometry = gpd.points_from_xy(df_aux_metadata['longitude'], df_aux_metadata['latitude']),
        crs = gdf_cluster.crs  # Match the CRS of the polygons
    )
    gdf_aux_metadata = gdf_aux_metadata.sjoin(gdf_cluster, how='inner', predicate='within')
    gdf_aux_metadata = gdf_aux_metadata[group_cols + ['codMunicipio', 'departamen', 'provincia', 'municipio']]

    df_target = gdf_aux_metadata.merge(
        df_target,
        how = 'left',
        on = group_cols
    )

    # Remove any missing coordinates from the population grid columns
    df_target = df_target.rename(columns={
        'longitude': 'lon_station',
        'latitude': 'lat_station'
    })
    df_target = df_target.dropna(subset=['lon', 'lat']).reset_index(drop=True)

    return df_target


#——————————————————————————————————————————————
# CREATE RAINFALL TARGET
#——————————————————————————————————————————————


def clean_prcp_station_data(
    df: pd.DataFrame,
):
    df_station = df.copy()

    var_replace = [
        'PRCP', 'prcpTru', 'prcpTruAccum', 'prcpGap', 'prcpGapLimit', 
        'weekCyc', 'weekCycStr', 'asyRoundPat', 'meaPrec', 
        'meaPrecInd', 'meaPrecInc', 'othQualProb', 'monOutlierAcmAll', 'monOutlierAcmSel'
    ]
    for var in var_replace:
        df_station[var] = df_station[var].mask(df_station[var]<=-99, np.nan)

    df_station['flagNA'] = np.where(
        ((df_station['PCUP']==1) & (df_station['SUCU2']==1) & (df_station['PCGA']==1)), 
        1, 0
    )
    df_station['flagNA_2'] = np.where(
        (df_station['prcpTru']>=2) |
        (df_station['prcpTruAccum']<=10) |
        (df_station['prcpGapLimit']>=4) |
        ((df_station['meaPrec']>=4) & 
         (df_station['othQualProb']>=3) & 
         (df_station['meaPrecInc']>=3) & 
         (df_station['meaPrecInd']>=3)) |
        ((df_station['monOutlierAcmAll']>=1) & (df_station['monOutlierAcmSel']>=1)),
        1, 0
    )

    df_station['PRCP_clean'] = np.where(
        (df_station['flagNA'] == 1) |  (df_station['flagNA_2'] == 1), 
        np.nan, df_station['PRCP']
    )
    return df_station


def process_prcp_station_data(
    df_var: pd.DataFrame, 
    params_t: dict, 
    params_s: dict,
    group_cols: list = ['decadeCode']
):
    """
    Creates a GeoDataframe with the information about the target variable for a subset of the grid
    Args:
        - ds_po : (xr.Dataset) Dataset containing the target population grid
        - df_tn : (pd.DataFrame) Dataframe with the time series of target variable
        - df_tn_metadata : (pd.DataFrame) Dataframe with the coordinates and other info about the weather stations
        - gdf_cluster : (pd.DataFrame) Dataframe with the geometries selected for our AOI
        - params_t : (dict) Dictionary with temporal parameters for slicing in time
        - params_target : (dict) Dictionary with spatial parameters for slicing in space
    Returns:
        pd.DataFrame with the target variable for a slice of the population target
    """
    start_date, end_date, fmt = params_t['start_date'], params_t['end_date'], params_t['date_format']
    start_date = pd.to_datetime(start_date, format=fmt)
    end_date = pd.to_datetime(end_date, format=fmt)

    # Clean data for in-situ stations
    #df_var = clean_prcp_station_data(df_var)
    df_var['PRCP_clean'] = df_var['PRCP']
    #df_var = subset_geometry(df_var, params_s)

    # Format data
    df_var = df_var.rename(columns={
        'Date': 'time',
    })
    df_var['time'] = pd.to_datetime(df_var['time'])
    
    # Create cluster
    df_var = add_time_coordinate_df(df_var, 'time', 'year')
    df_var = add_time_coordinate_df(df_var, 'time', 'week')

    coords = group_cols + ['latitude', 'longitude', 'altitude']
    df_weekly = df_var.groupby(coords + ['year', 'week'], as_index=False).agg(
        PRCP = ('PRCP', 'sum')
    )

    df_cluster = create_cluster(df_weekly, 'PRCP', agg_func='mean', group_cols=group_cols)
    df_var = df_var.merge(
        df_cluster[group_cols + ['cluster']],
        how = 'left',
        on = group_cols
    )

    # Select relevant columns
    df_var = df_var[
        group_cols + [ 
        'longitude',
        'latitude',
        'time', 
        'cluster',
        'PRCP'
    ]].copy()

    # Filter data within time frame and AOI
    df_var = df_var[
        (df_var['latitude'] > params_s['miny']) & (df_var['latitude'] < params_s['maxy']) &
        (df_var['longitude'] > params_s['minx']) & (df_var['longitude'] < params_s['maxx']) 
    ].copy()
    df_var = df_var[(df_var['time']>= start_date) & (df_var['time']<= end_date)].copy()

    return df_var


def get_window_days(doy, window=30):
    """Return list of days-of-year within ±window of doy (wrap around at 365)."""
    days = np.arange(doy - window, doy + window + 1)
    return ((days - 1) % 365) + 1


def get_percentile_climatology_df(
    df_orig: pd.DataFrame, 
    variable,
    group_cols: list = ['decadeCode'],
    clim_window = 7,
):
    """
    Creates a climatology of the selected variable grouping on the selected columns
    Args:
        -  df_orig: pd.DataFrame containing the original dataset with the variable to calculate climatology

    """
    # Check if dayof year is present as a coordinate:
    if 'dayofyear' not in df_orig.columns:
        raise ValueError('dayofyear is not present in the DataFrame to compute daily climatologies')
    
    df_target = df_orig.copy()
    df_list = []
    # For each day of year compute the climatology within a time window of days before and after
    for day in range(1, 366):
        list_days = get_window_days(day, window=clim_window)
        df_base = df_target[df_target['dayofyear'].isin(list_days)].copy()
        # For each percentile compute the climatology of the variable along the grouping columns
        df_base[f'perc_{variable}'] = df_base.groupby(group_cols)[variable].transform(
            lambda x: np.round(x.rank(method="average", pct=True)*100, 4)
        )
        df_base = df_base[
            df_base['dayofyear']==day
        ][
            group_cols + ['time', f'perc_{variable}']
        ].copy()
        df_list.append(df_base)

    df_full = pd.concat(df_list, axis=0, ignore_index=True)

    df_full = df_full.sort_values(by=group_cols+['time'])
    return df_full


def create_percentile_prcp_target(
    df_target,
    params_target,
    suffix
):
    #prcp_thresholds = params_target['prcp_thresholds']
    perc_thresholds = params_target['perc_thresholds']
    prcp_days = params_target['prcp_days']
    variable = params_target['variable']
    group_cols = params_target['group_cols']

    df_merge = df_target[group_cols + ['time', variable]].copy()

    df_merge[f'{variable}_bin_{suffix}'] = np.where(
        (df_merge[variable] > 0) & ~(df_merge[variable].isna()), 1, 0
    )

    # Add dayofyear coordinate to compute climatologies later
    df_merge = add_time_coordinate_df(df_merge, level='dayofyear')

    df_merge.sort_values(by= group_cols + ['time'], inplace=True)  
    for days in prcp_days:
        target_var = f'prcp_next{days}d_{suffix}'
        df_merge[target_var] = ( #f'prcp_last{days}d_{suffix}'
            df_merge
            .groupby(group_cols)[variable]
            .rolling(window=days, min_periods=1)
            .sum()
            .shift(-(days-1))
            .reset_index(level=0, drop=True)
        )
        # Get accumulated precipitation in the next d days
        #df_merge[target_var] = (
        #    df_merge
        #    .groupby(group_cols)[f'prcp_last{days}d_{suffix}']
        #    .shift(-(days-1))
        #)
        # Get percentiles of prcp in the next d days
        df_clim = get_percentile_climatology_df(
            df_merge,
            target_var,
            group_cols = group_cols,
            clim_window=3
        )
        df_merge = df_merge.merge(
            df_clim,
            how = 'left',
            on = group_cols + ['time']
        )
        # Create target if the percentile of the accumulated precipitacion in d days is above a threshold
        for perc in perc_thresholds:
            df_merge[f'target_P{perc}_{days}d_{suffix}'] = np.where(
                df_merge[f'perc_{target_var}'] >= perc, 1, 0
            )
        #df_merge.drop(columns=[f'prcp_last{days}d_{suffix}'], inplace=True)

    return df_merge


def create_extreme_rainfall_target( 
    df_prcp: pd.DataFrame, 
    ds_chirps: xr.Dataset,
    ds_po: xr.Dataset,
    gdf_cluster: gpd.GeoDataFrame, 
    params_t: dict, 
    params_target: dict,
    params_s: dict
):
    """
    Creates a GeoDataframe with the information about the target variable for a subset of the grid
    Args:
        - df_prcp : (pd.DataFrame) Dataframe with the time series of target variable
        - ds_chirps : (pd.DataFrame) Dataframe with the time series of precipitation from CHIRPS
        - ds_po : (xr.Dataset) Dataset containing the target population grid
        - gdf_cluster : (pd.DataFrame) Dataframe with the geometries selected for our AOI
        - params_t : (dict) Dictionary with temporal parameters for slicing in time
        - params_target : (dict) Dictionary with  parameters to build the target variables
        - params_s : (dict) Dictionary with spatial parameters for slicing in space
    Returns:
        pd.DataFrame with the target variable for a slice of the target population
    """
    start_date, end_date, freq, fmt = params_t['start_date'], params_t['end_date'], params_t['freq'], params_t['date_format']
    start_date = pd.to_datetime(start_date, format=fmt)
    end_date = pd.to_datetime(end_date, format=fmt)
    group_cols = params_target['group_cols']

    # Step 1: Create auxiliary dataframe with the coordinates and the grouping cols from weather stations
    df_prcp = subset_geometry(df_prcp, params_s)
    df_aux_metadata = df_prcp[group_cols + ['longitude', 'latitude']].drop_duplicates().copy()

    # Step 2: Compute the target variable options from weather station records
    # Clean weather station dataset
    df_clean = process_prcp_station_data(df_prcp, params_t, params_s, group_cols)
    
    # Fix time alignment, but keep original timestamp
    df_clean['time_orig'] = df_clean['time']
    df_clean['time'] = df_clean['time'] - pd.to_timedelta(1, unit='D')

    # Create a full cross-join dataset for all dates and locations
    df_dates = pd.DataFrame({
        'time': pd.date_range(start_date, end_date, freq=freq)
    })
    df_grid = df_clean[group_cols].drop_duplicates()
    df_grid = df_grid.merge(df_dates, how='cross')
    # Merge clean dataset with cross-join to fill in any gaps
    df_clean = df_grid.merge(
        df_clean,
        how = 'left',
        on = group_cols + ['time']
    ).rename(columns={'PRCP': 'prcp_lfa'})

    # Step 3: Compute the target variable options from CHIRPS
    # Match station coordinates to CHIRPS coordinates
    df_coords = match_grid_points(
        ds_chirps, 
        df_aux_metadata, 
        sample_coords=('lon', 'lat'), 
        target_coords=('longitude', 'latitude')
    )
    # Select coordinates from CHIRPS closest to the stations
    ds_slice = select_coordinates(ds_chirps, df_coords)
    ds_slice = ds_slice.reset_index('points')
    # Convert the slice of CHIRPS to a dataframe
    df_chirps = ds_slice.to_dataframe().reset_index().drop(columns=['band', 'spatial_ref'])
    # Append metadata to dataframe to compute the target by grouping columns
    df_chirps = df_chirps.merge(
        df_coords,
        how = 'left',
        on = ['lon', 'lat']
    ).rename(columns={'prcp': 'prcp_chirps'})
    df_chirps = df_chirps.drop(columns=['points', 'lon', 'lat'])
    # Filter dataset within time frame
    df_chirps = df_chirps[
        (df_chirps['time']>= start_date) 
        & (df_chirps['time']<= end_date)
    ].copy()

    # Step 4: Merge in-situ target dataset with CHIRPS target dataset
    df_target = df_aux_metadata.merge(
        df_clean,
        how = 'left',
        on = group_cols + ['longitude', 'latitude']
    ).merge(
        df_chirps,
        how = 'left',
        on = group_cols + ['longitude', 'latitude', 'time']
    )
   

    '''# Adjust daily prcp allocation from stations using CHIRPS 
    df_target['year'] = df_target['time'].dt.year
    df_target['week'] = df_target['time'].dt.isocalendar().week

    df_target['total_weekly_prcp'] = df_target.groupby(group_cols + ['year', 'week'])['prcp_chirps'].transform('sum')
    df_target['pct_prcp_chirps'] = np.where(
        df_target['total_weekly_prcp']==0, 0,
        df_target['prcp_chirps'] / df_target['total_weekly_prcp']
    )
    df_target['prcp_chirps_adj'] = (
        df_target['pct_prcp_chirps'] * 
        df_target.groupby(group_cols + ['year', 'week'])['prcp_lfa'].transform('sum')
    )
    '''
    # Adjust daily prcp allocation from stations using CHIRPS 
    df_target['year'] = df_target['time'].dt.year
    df_target['month'] = df_target['time'].dt.month
    # Compute monthly time series
    monthly_stats = (
        df_target.groupby(group_cols + ['year', 'month']).agg(
            chirps_monthly=('prcp_chirps', 'sum'),
            lfa_monthly=('prcp_lfa', 'sum'),
            day_count=('prcp_lfa', 'count'),
        ).reset_index()
        .query('day_count >= 26')
        .groupby(group_cols + ['month'])
        .agg(
            chirps_mean=('chirps_monthly', 'mean'),
            lfa_mean=('lfa_monthly', 'mean'),
        )
        .reset_index()
    )
    monthly_stats['bias_factor'] = np.where(
        monthly_stats['chirps_mean'] == 0, 1.0,
        monthly_stats['lfa_mean'] / monthly_stats['chirps_mean']
    )

    df_target = df_target.merge(
        monthly_stats,
        on=group_cols + ['month'],
        how='left',
    )
    df_target['bias_factor'] = df_target['bias_factor'].fillna(1.0) 
    df_target['prcp_chirps_adj'] = df_target['prcp_chirps'] * df_target['bias_factor']
    
    # Weighted average
    df_target['prcp'] = (
        0.5 * df_target['prcp_lfa'].fillna(0)
        + 0.5 * df_target['prcp_chirps_adj']
    )


    params_target['variable'] = 'prcp'
    df_target = df_target.drop(columns=['year', 'month'])
    df_target_aux = create_percentile_prcp_target(df_target, params_target, 'final')

    df_target = df_target.merge(
        df_target_aux.drop(columns=['prcp']),
        how = 'left',
        on = group_cols + ['time']
    )

    print('Target from gridded data succesfully created')   

    # Step 5: Add metadata (coordinates) from population grid
    df_coords_po = match_grid_points(
        ds_po, 
        df_aux_metadata, 
        sample_coords=('lon', 'lat'), 
        target_coords=('longitude', 'latitude')
    )
    df_target = df_coords_po.merge(
        df_target,
        how = 'left',
        on = group_cols + ['longitude', 'latitude']
    )
  
    # (Optional) Add metadata from geometry file
    gdf_aux_metadata = gpd.GeoDataFrame(
        df_aux_metadata,
        geometry = gpd.points_from_xy(df_aux_metadata['longitude'], df_aux_metadata['latitude']),
        crs = gdf_cluster.crs  # Match the CRS of the polygons
    )
    gdf_aux_metadata = gdf_aux_metadata.sjoin(gdf_cluster, how='inner', predicate='within')
    gdf_aux_metadata = gdf_aux_metadata[group_cols + ['codMunicipio', 'departamen', 'provincia', 'municipio']]

    df_target = gdf_aux_metadata.merge(
        df_target,
        how = 'left',
        on = group_cols
    )

    # Remove any missing coordinates from the population grid columns
    df_target = df_target.rename(columns={
        'longitude': 'lon_station',
        'latitude': 'lat_station'
    })
    df_target = df_target.dropna(subset=['lon', 'lat']).reset_index(drop=True)

    return df_target


#——————————————————————————————————————————————
# CREATE TARGET
#——————————————————————————————————————————————


def create_target(
    df_ground: pd.DataFrame, 
    ds_aux: xr.Dataset,
    ds_po: xr.Dataset,
    gdf_aoi: gpd.GeoDataFrame, 
    params_t: dict, 
    params_target: dict,
    params_s: dict
) -> pd.DataFrame:
    """
    Creates a GeoDataframe with the information about the target variable for a subset of the grid
    Args:
        - df_ground : (pd.DataFrame) Dataframe with the time series of target variable
        - ds_aux : (xr.Dataset) Dataset with auxiliary historical data of target variable
        - ds_po : (xr.Dataset) Dataset containing the target population grid
        - gdf_aoi : (pd.DataFrame) Dataframe with the geometries selected for our AOI
        - params_t : (dict) Dictionary with temporal parameters for slicing in time
        - params_target : (dict) Dictionary with  parameters to build the target variables
        - params_s : (dict) Dictionary with spatial parameters for slicing in space
    Returns:
        pd.DataFrame with the target variable for a slice of the target population
    """

    variable = params_target.get('variable', 'tmin')

    dict_target_fn = {
        'tmin': create_cold_spells_target,
        'prcp': create_extreme_rainfall_target,
    }

    df_target = dict_target_fn[variable](
        df_ground,
        ds_aux,
        ds_po,
        gdf_aoi,
        params_t,
        params_target,
        params_s
    )

    return df_target

