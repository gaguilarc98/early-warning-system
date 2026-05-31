from .utils import *


RENAME_DICT = {
    'swc': ['swc', 'swvl1', 'soil_water_content', 'sm', 'soil_moisture'],
    'tp_chirps': ['prcp', 'tp', 'total_precipitation', 'precipitation', 'precip', 'prc', 'pcp'],
    't2m': ['tmin', 't2m', '2t', 't', 'mn2t6', 'mn2t', 'mn2t24', 'minimum_2_m_temperature_in_the_last_6_hours'],
    'tmax': ['tmax', 't2m', '2t', 't', 'mx2t6', 'mx2t', 'mx2t24', 'maximum_2_m_temperature_in_the_last_6_hours'],
    'msl': ['msl', 'mean_sea_level_pressure'],
    'u10': ['10u', 'u10', '10_m_u_component_of_wind'],
    'v10': ['10v', 'v10', '10_m_v_component_of_wind'],
    'tp': ['tp', 'prcp', 'total_precipitation', 'precipitation', 'precip', 'prc', 'pcp'],
    'q': ['q', 'specific_humidity'],
    'altitude': ['altitude', 'height', 'h'],
}

#———————————————————————————————————————————
# RENAME VARIABLES
#———————————————————————————————————————————


def rename_vars(ds_orig, variable=None):
    """Rename variables in the original dataset with the registered names"""
    if variable is None:
        raise KeyError(f'{variable} is not in the dataset, a variable name must be provided.')
    ds = ds_orig.copy()

    # Get coordinate names
    lon, lat, time = get_coordinates(ds)

    # Get list of possible variable names
    if variable not in RENAME_DICT:
        raise KeyError(f'{variable} is not available in the registered variables in n02_process_data')

    list_vars = list(set(ds.data_vars).intersection(set(RENAME_DICT[variable])))
    if len(list_vars)==0:
        print(f'Either the variable name is not registered in n02_create_features or the variable is wrong.')
        raise KeyError(f'There is no variable in the dataset matching the requested variable: {variable}')

    var_name = list_vars[0]
    print(f'{var_name} was detected among the variables. It will be used and renamed to {variable}')

    dict_rename = {var_name: variable}
    if lon is not None:
        dict_rename[lon] = 'lon'
        print(f'Renaming {lon} to "lon"')
    if lat is not None:
        dict_rename[lat] = 'lat'
        print(f'Renaming {lat} to "lat"')
    if time is not None:
        dict_rename[time] = 'time'
        print(f'Renaming {time} to "time"')

    ds = ds.rename(dict_rename)

    return ds


#———————————————————————————————————————————
# AUXILIARY FUNCTIONS
#———————————————————————————————————————————


def slice_ds(ds, params):
    """Slice dataframe based on bounds and nodes by degree"""
    #TODO: Slice data based on grid yet TODO
    res_degree = 1/params['nodes_by_deg']
    
    # Make sure that longitude values are in the correct format and if not convert them
    lon, lat, time = get_coordinates(ds)
    ds[lon] = convert_lon_to_180(ds[lon].values)

    lons = np.arange(params['lon_min'], params['lon_max'] + res_degree, res_degree)
    lats = np.arange(params['lat_max'], params['lat_min'] - res_degree, -res_degree)
    ds = regrid_dataset(ds, lons=lons, lats=lats)
    
    return ds


def bound_time_grid(ds, params):
    """Slice dataset with temporal bounds"""
    # Get start and end dates of dataset
    start_date = params['start_date']
    end_date = params['end_date']
    # Get the names of coordinates
    lon, lat, time = get_coordinates(ds)
    #Slice the dataset with the temporal bounds
    ds = ds.sel(time = slice(start_date, end_date))
    return ds


#———————————————————————————————————————————
# DIGITAL ELEVATION FEATURES
#———————————————————————————————————————————


def compute_slope_aspect(ds_topo: xr.Dataset, var_elevation='altitude', z_factor=1.0, var = 'slope') -> xr.Dataset:
    """
    Calculate slope and aspect from an elevation DataArray using xarray's differentiate.

    Args:
        - elevation : (xr.DataArray) 2D elevation data with coordinate names (lat, lon) or (y, x).
        - z_factor : (float) Factor to convert vertical units to match horizontal 
            (e.g., 1 if units match; or to convert elevation meters to degrees).

    Returns:
        (xr.DataArray) DataArray with slope and aspect in degrees.
    """
    # Infer coordinate names
    xdim, ydim = tuple(ds_topo.sizes.keys())
    elevation = ds_topo[var_elevation]
    ycoord = elevation.coords[ydim]
    xcoord = elevation.coords[xdim]

    # Compute spacing in coordinate units
    dy = ycoord.differentiate(ydim)
    dx = xcoord.differentiate(xdim)

    # Convert spacing to 2D arrays for broadcasting
    dx2d = xr.broadcast(elevation, dx)[1]
    dy2d = xr.broadcast(elevation, dy)[0]

    # Compute gradients (dz/dx, dz/dy)
    dzdx = elevation.differentiate(xdim) / dx2d
    dzdy = elevation.differentiate(ydim) / dy2d

    # Apply z_factor (e.g., for degrees to meters conversion)
    dzdx = dzdx * z_factor
    dzdy = dzdy * z_factor

    # Slope: arctangent of gradient magnitude
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.rad2deg(slope_rad)

    # Aspect: compass direction of the slope
    aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = np.rad2deg(aspect_rad)
    aspect_deg = (90.0 - aspect_deg) % 360.0  # Convert to compass angle
    
    #TODO: Review the calculation of aspect as near no elevation could result in noisy angles
    #TODO: Plan on separating these functions as they could serve to create features from other variables
    if var=='slope':
        return xr.DataArray(slope_deg, coords=elevation.coords, dims=elevation.dims, name='slope')
    elif var=='aspect':
        return xr.DataArray(aspect_deg, coords=elevation.coords, dims=elevation.dims, name='aspect')


def compute_slope(
    ds_topo: xr.Dataset,
) -> xr.DataArray:
    ds = ds_topo['altitude']
    ds = slope(ds)
    return ds

def compute_aspect(
    ds_topo: xr.Dataset,
) -> xr.DataArray:
    ds = ds_topo['altitude']
    ds = aspect(ds)
    return ds


def compute_tpi(ds_topo: xr.Dataset, var_elevation = 'altitude', window_size: int = 5) -> xr.Dataset:
    """
    Compute Topographic Position Index (TPI) from an elevation DataArray.
    Args:
        - elevation : xr.DataArray 2D array of elevation values.
        - window_size : int Size of the moving window (must be odd).

    Returns:
        xr.Dataset Dataset containing TPI and TDI as DataArrays.
    """
    # Infer coordinate names
    elevation = ds_topo[var_elevation]
    # Ensure window size is odd
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")

    # Convert xarray DataArray to numpy array for processing
    elevation_np = elevation.values

    # Compute mean elevation in the neighborhood
    mean_elev = uniform_filter(elevation_np, size=window_size, mode='nearest')

    # Compute TPI
    tpi = elevation_np - mean_elev

    # Create DataArrays for TPI
    return xr.DataArray(tpi, coords=elevation.coords, dims=elevation.dims, name=f'TPI_{window_size}')


def compute_tdi(ds_topo: xr.Dataset, var_elevation = 'altitude', window_size: int = 5) -> xr.Dataset:
    """
    Compute Topographic Dissection Index (TDI) from an elevation DataArray.
    Args:
        - elevation : xr.DataArray 2D array of elevation values.
        - window_size : int Size of the moving window (must be odd).
    Returns:
        xr.Dataset Dataset containing TPI and TDI as DataArrays.
    """
    # Infer coordinate names
    elevation = ds_topo[var_elevation]
    # Ensure window size is odd
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")
    
    # Convert xarray DataArray to numpy array for processing
    elevation_np = elevation.values

    # Compute min and max and mean elevation in the neighborhood
    min_elev = minimum_filter(elevation_np, size=window_size, mode='nearest')
    max_elev = maximum_filter(elevation_np, size=window_size, mode='nearest')
    mean_elev = uniform_filter(elevation_np, size=window_size, mode='nearest')

    # Compute TDI while avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tdi = np.where(mean_elev != 0, (max_elev - min_elev) / mean_elev, 0)

    return xr.DataArray(tdi, coords=elevation.coords, dims=elevation.dims, name=f'TDI_{window_size}')


def create_features_topo(
    ds_po: xr.Dataset, 
    ds_topo: xr.Dataset, 
    params: dict,
    gdf_target: gpd.GeoDataFrame = None
) -> xr.Dataset:
    """
    Create features from a Digital Elevation Model (DEM)

    Args:
        - ds_po : (xr.Dataset) Dataset of target population to slice data
        - ds_topo : (xr.Dataset) Dataset contatining altitude as a variable
        - params : (dict) Dictionary containing parameters such as z_factor, window_sizes for TPI and TDI
            window_tpi: Odd number of nodes for window to calculate Topographic Position Index
            window_tdi: Odd number of nodes for window to calculate Terrain Diversity Index
        - gdf_target : (gpd.GeoDataFrame) DataFrame containing aditional slicing for station locations
    
    Returns:
        xr.Dataset Dataset with topographic features
    """
    ds = ds_topo.squeeze()
    ds = rename_vars(ds, 'altitude')
    ds = ds[['altitude']]
    ds = remove_single_coordinates(ds)

    lon, lat, time = get_coordinates(ds)
    ds[lon] = convert_lon_to_180(ds[lon].values)

    ds = regrid_dataset(ds, lons=ds_po['lon'].values, lats=ds_po['lat'].values)
    
    # Add slope and aspect
    ds['slope'] = compute_slope_aspect(ds, 'altitude', z_factor=1/111000, var='slope')
    #ds['aspect'] = compute_slope_aspect(ds, 'altitude', z_factor=1/111000, var='aspect')
    #ds['slope'] = compute_slope(ds_topo)
    ds['aspect'] = compute_aspect(ds)
    
    # Add TPI features using the specified windows
    for w in params['windows_tpi']:
        ds[f'TPI_{w}'] = compute_tpi(ds, 'altitude', window_size=w)
    
    # Add TDI features using the specified windows
    for w in params['windows_tpi']:
        ds[f'TDI_{w}'] = compute_tdi(ds, 'altitude', window_size=w)
    
    # Add latitude and longitude as 1D variables
    ds = ds.assign(
        longitude=('lon', ds['lon'].values), 
        latitude=('lat', ds['lat'].values)
    )

    if gdf_target is None:
        return ds
    else:
        ds_slice = select_coordinates(ds, gdf_target)
        ds_slice = ds_slice.reset_index('points')
        return ds_slice



#———————————————————————————————————————————
# FORECAST FEATURES
#———————————————————————————————————————————


def add_climatology_lazy(ds, ds_clim, field, var_name='climatology', level = 'week'):
    '''Adds a pre-computed climatology to the original dataset'''
    lookup = ds[level]
    clim_values = ds_clim[field].sel({level: lookup})
    ds[var_name] = clim_values

    return ds


def get_climatology(ds, field, var_name='climatology', time_dim='time', level='dayofyear', smooth_window=0, func='mean'):
    '''Get climatologies for the specified variable at the level selected'''
    ds = add_time_coordinate(ds, time_dim, level)

    # Only keep full years
    year = ds[time_dim].dt.year
    year_counts = year.groupby(year).count()
    full_years = year_counts.where(year_counts >= 360, drop=True).coords[year.name]

    ds_full = ds.sel({time_dim: ds[time_dim].dt.year.isin(full_years)})

    # Lazy climatology
    if func == 'mean':
        ds_clim = ds_full[[field]].groupby(level).mean(dim=time_dim)
    elif func == 'std':
        ds_clim = ds_full[[field]].groupby(level).std(dim=time_dim)
    if smooth_window > 0:
        ds_clim = xr.concat([
            ds_clim.isel({level: slice(-smooth_window, None)}), 
            ds_clim, 
            ds_clim.isel({level: slice(None, smooth_window)})
        ], dim=level)
        
        ds_clim = ds_clim.rolling({level: 2*smooth_window+1}, center=True, min_periods=1).mean()
        ds_clim = ds_clim.isel({level: slice(smooth_window, -smooth_window)})

    dict_chunk = {}
    for dim in ds_clim.dims:
        dict_chunk[dim] = -1
    ds_clim = ds_clim.chunk(dict_chunk)

    # Add interpolated climatology (map week to week)
    ds = add_climatology_lazy(ds, ds_clim, field, var_name, level)
    
    ds_clim = ds_clim.rename({field: var_name})

    return ds, ds_clim 


def rename_vars_dict(ds, dict_vars):
    dict_rename = {}
    for key in dict_vars:
        if dict_vars[key] is None:
            continue
        else:
            for var, name in dict_vars[key].items():
                if f'{name}' in ds.data_vars:
                    dict_rename[f'{name}'] = f'{key}_{var}'
    
    ds = ds.rename(dict_rename)
    return ds, dict_rename


PRECIP_FORECAST_VARS = {'tp', 'prcp'}
PRECIP_CHIRPS_VARS   = {'tp_chirps', 'prcp_chirps'}
 
 
def create_features_forecast(
    ds_fct: xr.Dataset,
    params_fct: dict,
    params_t: dict,
) -> xr.Dataset:
    """
    Create features from weather forecasts.
 
    Args:
        ds_fct     : dataset containing forecasted variables.
        params_fct : keys 'variable', 'steps', 'smooth_clim'.
        params_t   : temporal parameters for date range.
 
    Returns:
        ds_fct       : xr.Dataset of forecast features.
        ds_clim_full : xr.Dataset of climatology and std variables.
    """
    variable    = params_fct['variable']
    steps       = params_fct['steps']
    smooth_clim = params_fct.get('smooth_clim', 2)
 
    date_range = get_date_array(params_t)
 
    # Rename variable and coordinates to canonical names, subset immediately
    ds_fct = rename_vars(ds_fct, variable=variable)
    ds_fct = ds_fct[[variable]].copy()
 
    # Convert longitude range
    ds_fct['lon'] = convert_lon_to_180(ds_fct['lon'])
 
    # Remove unnecessary coordinates
    vars_rm = set(['reftime', 'valid_time']).intersection(set(ds_fct.coords))
    if vars_rm:
        ds_fct = ds_fct.drop_vars(vars_rm)
 
    # Step handling depends on variable type
    if variable in PRECIP_FORECAST_VARS:
        # Accumulated precipitation: difference between max and min step
        ds_fct['step'] = ds_fct['step'].dt.total_seconds() / 3600 / 24
        ds_fct['time'] = ds_fct['time'] + pd.to_timedelta(min(steps), unit='days')
        ds_fct['time'] = pd.to_datetime(ds_fct['time']).normalize()
        ds_fct = ds_fct.sel(step=max(steps)) - ds_fct.sel(step=min(steps))
        keep_coords = ['time', 'lat', 'lon']
 
    elif variable in PRECIP_CHIRPS_VARS:
        # CHIRPS: no step dimension
        ds_fct['time'] = pd.to_datetime(ds_fct['time']).normalize()
        keep_coords = ['time', 'lat', 'lon']
 
    else:
        # Standard forecast: select steps and keep step coordinate
        ds_fct['step'] = ds_fct['step'].dt.total_seconds() / 3600 / 24
        ds_fct = ds_fct.sel(step=steps)
        ds_fct['time'] = ds_fct['time'] + pd.to_timedelta(min(steps), unit='days')
        ds_fct['time'] = pd.to_datetime(ds_fct['time']).normalize()
        keep_coords = ['time', 'lat', 'lon', 'step']
 
    # Drop single-value coordinates
    ds_fct = drop_single_coords(ds_fct, except_coords=keep_coords)
 
    # Subset to requested date range
    ds_fct = ds_fct.sel(time=ds_fct['time'][ds_fct['time'].isin(date_range)])
 
    # Compute climatology, std, and anomaly
    ds_clim_full = []
    ds_fct, ds_clim = get_climatology(ds_fct, variable, f'clim_{variable}', level='week', smooth_window=smooth_clim, func='mean')
    ds_fct, ds_std  = get_climatology(ds_fct, variable, f'std_{variable}',  level='week', smooth_window=smooth_clim, func='std')
    ds_fct[f'anom_{variable}'] = (ds_fct[variable] - ds_fct[f'clim_{variable}']) / ds_fct[f'std_{variable}']
    ds_fct = ds_fct.drop_vars([f'clim_{variable}', f'std_{variable}'])
    ds_clim_full.append(ds_clim)
    ds_clim_full.append(ds_std)
 
    ds_fct = ds_fct.drop_vars('week')
    ds_clim_full = xr.merge(ds_clim_full)
 
    return ds_fct, ds_clim_full


def create_features_precip(
    #ds_po: xr.Dataset, 
    ds_fct: xr.Dataset, 
    params: dict,
    params_t: dict,
    #gdf_target: gpd.GeoDataFrame = None,
) -> xr.Dataset:
    """"
    Creates features from weather forecasts
    Args:
        - ds_po : (xr.Dataset) Dataset containing the grid for the target population
        - ds_fct : (xr.Dataset) Dataset contating forecasted variables
        - params : (dict) Dictionary containing parameters for forecast variables
        - gdf_target : (gpd.GeoDataFrame) Dataframe containing station locations to slice on
    Returns:
        xr.Dataset Dataset of features from weather forecast
    """
    # Get parameters from dictionary
    dict_vars = params['dict_vars']
    steps = params['steps']
    #fct_step = params['step']
    #next_step = params['next_step']
    #time_align = params['time_align']
    smooth_clim = params['smooth_clim']
    
    """
    # Slice grid dataset with temporal bounds
    ds_po = bound_time_grid(ds_po, params_t)
    """
    date_range = get_date_array(params_t)

    # Formating latitude, longitude and time coordinates
    ds_fct = ds_fct.squeeze()
    lon, lat, time = get_coordinates(ds_fct)
    ds_fct[lon] = convert_lon_to_180(ds_fct[lon])
    dict_rename = {
        lon: 'lon',
        lat: 'lat',
    }
    if time is not None:
        dict_rename[time] = 'time'
    ds_fct = ds_fct.rename(dict_rename)

    # Remove unnecesary coordinates
    vars_rm = set(['reftime', 'valid_time']).intersection(set(ds_fct.coords))
    if len(vars_rm)>0:
        ds_fct = ds_fct.drop_vars(vars_rm)

    # Select a time step for forecast and relabel the time coordinate
    ds_fct['step'] = ds_fct['step'].dt.total_seconds()/3600/24
    #ds_fct = ds_fct.sel(step=steps)
    ds_fct['time'] = ds_fct['time'] + pd.to_timedelta(min(steps), unit='days')
    ds_fct['time'] = pd.to_datetime(ds_fct['time']).normalize()

    ds_max = ds_fct.sel(step=max(steps))
    ds_fct = ds_fct.sel(step=min(steps))
    ds_fct = ds_max - ds_fct

    # Remove unnecesary coordinates
    keep_coords = ['time', 'lat', 'lon'] # list of coords to keep
    ds_fct = drop_single_coords(ds_fct, except_coords=keep_coords)

    # Create accumulated variables
    #list_new_vars = []
    #for value  in dict_names.values():
    #    new_name = f'{value}_cum'
    #    ds_fct[new_name] = (ds_max[value] - ds_fct[value])
    #    ds_fct[new_name] = ds_fct[new_name].clip(min=0) # Clip negative values
    #    list_new_vars.append(new_name)

    # Regrid dataset to match the target resolution
    ds_fct = ds_fct.sel(
        time = ds_fct['time'][ds_fct['time'].isin(date_range)]
    )

    ds_fct, dict_names = rename_vars_dict(ds_fct, dict_vars)
    ds_fct = ds_fct[list(dict_names.values())].copy()

    # Compute climatologies and anomalies
    #ds_fct = ds_fct[list_new_vars]
    ds_clim_full = []
    for var in list(dict_names.values()):
        ds_fct, ds_clim = get_climatology(ds_fct, var, f'clim_{var}', level='week', smooth_window=smooth_clim, func='mean')
        ds_fct, ds_std = get_climatology(ds_fct, var, f'std_{var}', level='week', smooth_window=smooth_clim, func='std')
        ds_fct[f'anom_{var}'] = (ds_fct[var] - ds_fct[f'clim_{var}'])/ds_fct[f'std_{var}']
        
        ds_fct = ds_fct.drop_vars([f'clim_{var}', f'std_{var}'])

        ds_clim_full.append(ds_clim)
        ds_clim_full.append(ds_std)

    ds_fct = ds_fct.drop_vars('week')
    ds_clim_full = xr.merge(ds_clim_full)

    #if gdf_target is None:
    #    return ds_fct, ds_clim_full
    #else:
    #    ds_slice = select_coordinates(ds_fct, gdf_target)
    #    ds_slice = ds_slice.reset_index('points')
    #    return ds_slice, ds_clim_full

    return ds_fct, ds_clim_full
    

def create_features_precip_chirps(
    #ds_po: xr.Dataset, 
    ds_fct: xr.Dataset, 
    params: dict,
    params_t: dict,
    #gdf_target: gpd.GeoDataFrame = None,
) -> xr.Dataset:
    """"
    Creates features from weather forecasts
    Args:
        - ds_po : (xr.Dataset) Dataset containing the grid for the target population
        - ds_fct : (xr.Dataset) Dataset contating forecasted variables
        - params : (dict) Dictionary containing parameters for forecast variables
        - gdf_target : (gpd.GeoDataFrame) Dataframe containing station locations to slice on
    Returns:
        xr.Dataset Dataset of features from weather forecast
    """
    # Get parameters from dictionary
    dict_vars = params['dict_vars']
    steps = params['steps']
    #fct_step = params['step']
    #next_step = params['next_step']
    #time_align = params['time_align']
    smooth_clim = params['smooth_clim']
    
    """
    # Slice grid dataset with temporal bounds
    ds_po = bound_time_grid(ds_po, params_t)
    """
    date_range = get_date_array(params_t)

    # Formating latitude, longitude and time coordinates
    ds_fct = ds_fct.squeeze()
    lon, lat, time = get_coordinates(ds_fct)
    ds_fct[lon] = convert_lon_to_180(ds_fct[lon])
    dict_rename = {
        lon: 'lon',
        lat: 'lat',
    }
    if time is not None:
        dict_rename[time] = 'time'
    ds_fct = ds_fct.rename(dict_rename)

    # Remove unnecesary coordinates
    vars_rm = set(['reftime', 'valid_time']).intersection(set(ds_fct.coords))
    if len(vars_rm)>0:
        ds_fct = ds_fct.drop_vars(vars_rm)

    # Select a time step for forecast and relabel the time coordinate
    #ds_fct['step'] = ds_fct['step'].dt.total_seconds()/3600/24
    #ds_fct = ds_fct.sel(step=steps)
    #ds_fct['time'] = ds_fct['time'] + pd.to_timedelta(min(steps), unit='days')
    ds_fct['time'] = pd.to_datetime(ds_fct['time']).normalize()

    #ds_fct = ds_fct.sum(dim='step')

    # Remove unnecesary coordinates
    keep_coords = ['time', 'lat', 'lon'] # list of coords to keep
    ds_fct = drop_single_coords(ds_fct, except_coords=keep_coords)

    # Create accumulated variables
    #list_new_vars = []
    #for value  in dict_names.values():
    #    new_name = f'{value}_cum'
    #    ds_fct[new_name] = (ds_max[value] - ds_fct[value])
    #    ds_fct[new_name] = ds_fct[new_name].clip(min=0) # Clip negative values
    #    list_new_vars.append(new_name)

    # Regrid dataset to match the target resolution
    ds_fct = ds_fct.sel(
        time = ds_fct['time'][ds_fct['time'].isin(date_range)]
    )

    ds_fct, dict_names = rename_vars_dict(ds_fct, dict_vars)
    ds_fct = ds_fct[list(dict_names.values())].copy()

    # Compute climatologies and anomalies
    #ds_fct = ds_fct[list_new_vars]
    ds_clim_full = []
    for var in list(dict_names.values()):
        ds_fct, ds_clim = get_climatology(ds_fct, var, f'clim_{var}', level='week', smooth_window=smooth_clim, func='mean')
        ds_fct, ds_std = get_climatology(ds_fct, var, f'std_{var}', level='week', smooth_window=smooth_clim, func='std')
        ds_fct[f'anom_{var}'] = (ds_fct[var] - ds_fct[f'clim_{var}'])/ds_fct[f'std_{var}']

        ds_fct = ds_fct.drop_vars([f'clim_{var}', f'std_{var}'])
        
        ds_clim_full.append(ds_clim)
        ds_clim_full.append(ds_std)

    ds_fct = ds_fct.drop_vars('week')
    ds_clim_full = xr.merge(ds_clim_full)

    #if gdf_target is None:
    #    return ds_fct, ds_clim_full
    #else:
    #    ds_slice = select_coordinates(ds_fct, gdf_target)
    #    ds_slice = ds_slice.reset_index('points')
    #    return ds_slice, ds_clim_full

    return ds_fct, ds_clim_full
    

########____CLIMATOLOGIES DOMAIN____########


def create_features_fct_clim(
    ds_fct: xr.Dataset,
    ds_clim: xr.Dataset,
    params_fct: dict,
    params_t: dict,
) -> xr.Dataset:
    """
    Create forecast features using a precomputed climatology.

    Args:
        ds_fct     : dataset containing forecasted variables.
        ds_clim    : precomputed climatology and std (output of create_features_forecast).
        params_fct : keys 'variable', 'steps'.
        params_t   : temporal parameters for date range.

    Returns:
        ds_fct : xr.Dataset of forecast features with anomalies.
    """
    variable = params_fct['variable']
    steps    = params_fct['steps']

    date_range = get_date_array(params_t)

    # Rename variable and coordinates to canonical names, subset immediately
    if variable in PRECIP_FORECAST_VARS | PRECIP_CHIRPS_VARS:
        ds_fct = ds_fct.squeeze()
    ds_fct = rename_vars(ds_fct, variable=variable)
    ds_fct = ds_fct[[variable]].copy()

    # Convert longitude range
    ds_fct['lon'] = convert_lon_to_180(ds_fct['lon'])

    # Remove unnecessary coordinates
    vars_rm = set(['reftime', 'valid_time']).intersection(set(ds_fct.coords))
    if vars_rm:
        ds_fct = ds_fct.drop_vars(vars_rm)

    # Step handling depends on variable type
    if variable in PRECIP_FORECAST_VARS:
        ds_fct['step'] = ds_fct['step'].dt.total_seconds() / 3600 / 24
        ds_fct['time'] = ds_fct['time'] + pd.to_timedelta(min(steps), unit='days')
        ds_fct['time'] = pd.to_datetime(ds_fct['time'].values).normalize()
        ds_fct = ds_fct.sel(step=max(steps)) - ds_fct.sel(step=min(steps))
        keep_coords = ['time', 'lat', 'lon']

    elif variable in PRECIP_CHIRPS_VARS:
        ds_fct['time'] = pd.to_datetime(ds_fct['time'].values).normalize()
        keep_coords = ['time', 'lat', 'lon']

    else:
        ds_fct['step'] = ds_fct['step'].dt.total_seconds() / 3600 / 24
        ds_fct = ds_fct.sel(step=steps)
        ds_fct['time'] = ds_fct['time'] + pd.to_timedelta(min(steps), unit='days')
        ds_fct['time'] = pd.to_datetime(ds_fct['time'].values).normalize()
        keep_coords = ['time', 'lat', 'lon', 'step']

    # Drop single-value coordinates and duplicate times
    ds_fct = drop_single_coords(ds_fct, except_coords=keep_coords)
    if 'time' not in ds_fct.dims:
        ds_fct = ds_fct.expand_dims('time')
    ds_fct = ds_fct.drop_duplicates(dim='time')

    # Subset to requested date range
    ds_fct = ds_fct.sel(time=ds_fct['time'][ds_fct['time'].isin(date_range)])

    # Compute anomaly using precomputed climatology
    ds_fct = add_time_coordinate(ds_fct)
    ds_fct = add_climatology_lazy(ds_fct, ds_clim, f'clim_{variable}', f'clim_{variable}', level='week')
    ds_fct = add_climatology_lazy(ds_fct, ds_clim, f'std_{variable}',  f'std_{variable}',  level='week')

    ds_fct[f'anom_{variable}'] = (ds_fct[variable] - ds_fct[f'clim_{variable}']) / ds_fct[f'std_{variable}']
    ds_fct = ds_fct.drop_vars([f'clim_{variable}', f'std_{variable}'])

    ds_fct = ds_fct.drop_vars('week')
    return ds_fct
    

def create_features_clim_precip(
    ds_po: xr.Dataset, 
    ds_fct: xr.Dataset, 
    ds_clim: xr.Dataset, 
    params: dict,
    params_t: dict,
    gdf_target: gpd.GeoDataFrame = None,
) -> xr.Dataset:
    """"
    Creates features from weather forecasts
    Args:
        - ds_po : (xr.Dataset) Dataset containing the grid for the target population
        - ds_fct : (xr.Dataset) Dataset contating forecasted variables
        - params : (dict) Dictionary containing parameters for forecast variables
        - gdf_target : (gpd.GeoDataFrame) Dataframe containing station locations to slice on
    Returns:
        xr.Dataset Dataset of features from weather forecast
    """
    # Get parameters from dictionary
    dict_vars = params['dict_vars']
    fct_step = params['step']
    next_step = params['next_step']
    time_align = params['time_align']
    smooth_clim = params['smooth_clim']

    '''
    # Slice grid dataset with temporal bounds
    ds_po = bound_time_grid(ds_po, params_t)
    '''
    date_range = get_date_array(params_t)

    # Formating latitude, longitude and time coordinates
    ds_fct = ds_fct.squeeze()
    lon, lat, time = get_coordinates(ds_fct)
    ds_fct[lon] = convert_lon_to_180(ds_fct[lon])
    dict_rename = {
        lon: 'lon',
        lat: 'lat',
    }
    if time is not None:
        dict_rename[time] = 'time'
    ds_fct = ds_fct.rename(dict_rename)

    # Remove unnecesary coordinates
    vars_rm = set(['reftime', 'valid_time']).intersection(set(ds_fct.coords))
    if len(vars_rm)>0:
        ds_fct = ds_fct.drop_vars(vars_rm)

    # Select a time step for forecast and relabel the time coordinate
    ds_fct['step'] = ds_fct['step'].dt.total_seconds()/3600
    ds_fct['time'] = ds_fct['time'] + pd.to_timedelta(time_align, unit='h')

    ds_fct, dict_names = rename_vars_dict(ds_fct, dict_vars)
    ds_fct = ds_fct[list(dict_names.values())].copy()

    ds_aux = ds_fct.sel(step=next_step)
    ds_fct = ds_fct.sel(step=fct_step)

    # Create accumulated variables
    list_new_vars = []
    for value  in dict_names.values():
        new_name = f'{value}_cum'
        ds_fct[new_name] = (ds_aux[value] - ds_fct[value])
        ds_fct[new_name] = ds_fct[new_name].clip(min=0) # Clip negative values
        list_new_vars.append(new_name)

    # Remove any coordinates with a single value
    ds_fct = remove_single_coordinates(ds_fct)

    # Regrid dataset to match the target resolution
    ds_fct = regrid_dataset(ds_fct, lons=ds_po['lon'], lats=ds_po['lat'])
    ds_fct = ds_fct.sel(
        time = ds_fct['time'][ds_fct['time'].isin(date_range)]
    )

    # Compute climatologies and anomalies
    ds_fct = ds_fct[list_new_vars]
    for var in list_new_vars:
        ds_fct = add_time_coordinate(ds_fct)
        ds_fct = add_climatology_lazy(ds_fct, ds_clim, var, f'clim_{var}', level='week')

        ds_fct[f'anom_{var}'] = ds_fct[var] - ds_fct[f'clim_{var}']

    if gdf_target is None:
        return ds_fct
    else:
        ds_slice = select_coordinates(ds_fct, gdf_target)
        ds_slice = ds_slice.reset_index('points')
        return ds_slice