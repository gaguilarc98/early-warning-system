from .utils import *

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .a01_aoi_period import * #BoundingBox, get_bounds, resolve_period, describe_period, get_year_list
from .a01_data_downloaders import * #DataDownloader, DataConfig, ERA5Downloader, ERA5Config, UCSBDownloader, UCSBConfig


# ──────────────────────────────────────────────
# PROVIDER REGISTRY
# ──────────────────────────────────────────────

@dataclass
class ProviderConfig:
    provider: str
    downloader_cls: type[DataDownloader]
    config_cls: type[DataConfig]

    def __post_init__(self):
        if not issubclass(self.downloader_cls, DataDownloader):
            raise TypeError(f"{self.downloader_cls} must be a subclass of DataDownloader")
        if not issubclass(self.config_cls, DataConfig):
            raise TypeError(f"{self.config_cls} must be a subclass of DataConfig")


PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "ERA5":  ProviderConfig("ERA5",  ERA5Downloader, ERA5Config),
    "UCSB":  ProviderConfig("UCSB",  UCSBDownloader, UCSBConfig),
    "ECMWF": ProviderConfig("ECMWF", ERA5Downloader, ERA5Config),
}


def get_provider_config(provider: str) -> ProviderConfig:
    key = provider.upper().replace(" ", "_").replace("-", "_")
    if key not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider '{provider}'. Valid options: {list(PROVIDER_REGISTRY.keys())}")
    return PROVIDER_REGISTRY[key]


# ──────────────────────────────────────────────
# DEFAULT FIELD PARAMETERS REGISTRY
# ──────────────────────────────────────────────

FIELD_DEFAULTS: dict[tuple, dict] = {
    ("UCSB", "prcp"): dict(
        origin="CHIRPS-v3-ERA5",
        variable="PRCP",
        freq="daily",
    ),
    ("UCSB", "tmin"): dict(
        origin="CHIRTS-ERA5",
        variable="TN",
        freq="daily",
    ),
    ("UCSB", "tmax"): dict(
        origin="CHIRTS-ERA5",
        variable="TX",
        freq="daily",
    ),
    ("ERA5", "swc"): dict(
        product_type="reanalysis-era5-land",
        variable=["volumetric_soil_water_layer_1"],
        time=["08"],
    ),
    ("ERA5", "prcp"): dict(
        product_type="reanalysis-era5-land",
        variable=["total_precipitation"],
        time=["00"],
    ),
    ("ERA5", "tmin"): dict(
        product_type="reanalysis-era5-single-levels",
        variable=["minimum_2m_temperature_since_previous_post_processing"],
        time=["06"],
    ),
    ("ERA5", "tmax"): dict(
        product_type="reanalysis-era5-single-levels",
        variable=["maximum_2m_temperature_since_previous_post_processing"],
        time=["18"],
    ),
    ("ECMWF", "10u"): dict(
        product_type="tigge-forecasts",
        variable=["10_m_u_component_of_wind"],
        time=["00"],
        data_format="grib",
    ),
    ("ECMWF", "10v"): dict(
        product_type="tigge-forecasts",
        variable=["10_m_v_component_of_wind"],
        time=["00"],
        data_format="grib",
    ),
    ("ECMWF", "tmin"): dict(
        product_type="tigge-forecasts",
        variable=["minimum_2_m_temperature_in_the_last_6_hours"],
        time=["12"],
        data_format="grib",
    ),
    ("ECMWF", "msl"): dict(
        product_type="tigge-forecasts",
        variable=["mean_sea_level_pressure"],
        time=["00"],
        data_format="grib",
    ),
    ("ECMWF", "swc"): dict(
        product_type="tigge-forecasts",
        variable=["soil_moisture"],
        time=["00"],
        data_format="grib",
    ),
    ("ECMWF", "tp"): dict(
        product_type="tigge-forecasts",
        variable=["total_precipitation"],
        time=["00"],
        data_format="grib",
    ),
    ("ECMWF", "q"): dict(
        product_type="tigge-forecasts",
        variable=["specific_humidity"],
        time=["00"],
        data_format="grib",
    ),
}


def get_field_defaults(provider: str, field: str) -> dict:
    """Return default params for a (provider, field) pair. Returns empty dict if not registered."""
    key = (
        provider.upper().replace("-", "_"),
        field.lower(),
    )
    if key not in FIELD_DEFAULTS:
        print(f"No defaults registered for (provider, field) = {key}. Proceeding with user params only.")
        return {}
    return FIELD_DEFAULTS[key].copy()


# ──────────────────────────────────────────────
# REQUEST CONTAINER
# ──────────────────────────────────────────────

@dataclass
class DataRequest:
    """Fully resolved download request. Construct via DataRequest.build()."""
    provider: str
    field: str
    aoi: BoundingBox
    start_year: int
    end_year: int
    start_date: str
    end_date: str
    date_mode: bool
    params_request: dict
    #meta: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        params_request: dict,
        params_s: dict,
        params_t: Optional[dict] = None,
    ) -> "DataRequest":
        """
        Build and validate a DataRequest from input dicts.

        Args:
            params_request : must include 'provider' and 'field'; other keys override registered defaults.
            params_s       : resolved BoundingBox dict (output of extract_aoi()).
            params_t       : keys 'start_date', 'end_date' (YYYY-MM-DD); optional 'date_mode' (bool).

        Returns:
            DataRequest
        """
        params_request = params_request.copy()
        params_t       = (params_t or {})

        provider   = params_request.get("provider")
        field_name = params_request.get("field")
        if not provider:
            raise ValueError("params_request must include 'provider'.")
        if not field_name:
            raise ValueError("params_request must include 'field'.")

        provider   = provider.upper().replace("-", "_")
        field_name = field_name.lower()

        get_provider_config(provider)

        defaults = get_field_defaults(provider, field_name)
        merged   = defaults | params_request

        aoi = BoundingBox.from_dict(params_s)

        start_date = params_t.get("start_date")
        end_date   = params_t.get("end_date")
        date_mode  = bool(params_t.get("date_mode", False))

        if start_date is None or end_date is None:
            raise ValueError("params_t must include 'start_date' and 'end_date' (YYYY-MM-DD).")

        start_year, end_year = resolve_period(
            start_year=int(start_date[:4]),
            end_year=int(end_date[:4]),
        )

        #meta = {
        #    k: merged.pop(k)
        #    for k in ("country",)
        #    if k in merged
        #}

        return cls(
            provider=provider,
            field=field_name,
            aoi=aoi,
            start_year=start_year,
            end_year=end_year,
            start_date=start_date,
            end_date=end_date,
            date_mode=date_mode,
            params_request=merged,
            #meta=meta,
        )

    def build_downloader(self) -> DataDownloader:
        """Instantiate the correct downloader from the provider registry."""
        provider_cfg = get_provider_config(self.provider)
        config       = provider_cfg.config_cls.from_dict(self.params_request)
        return provider_cfg.downloader_cls(aoi=self.aoi, params_request=config)

    @property
    def years(self) -> List[int]:
        return get_year_list(self.start_year, self.end_year)

    @property
    def range_key(self) -> str:
        """Result key for date_mode. Format: YYYYMMDD_YYYYMMDD."""
        s = pd.to_datetime(self.start_date).strftime('%Y%m%d')
        e = pd.to_datetime(self.end_date).strftime('%Y%m%d')
        return f"{s}_{e}"

    def describe(self) -> str:
        if self.date_mode:
            return (
                f"provider={self.provider} | field={self.field} | "
                f"date_mode=True | range={self.range_key}"
            )
        return (
            f"provider={self.provider} | field={self.field} | "
            f"period={describe_period(self.start_year, self.end_year)}"
        )


@dataclass
class DataRequestResult:
    """Output container for a completed DataRequest."""
    request: DataRequest
    data: Dict[str, xr.Dataset] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.data) and not self.errors

    @property
    def partial(self) -> bool:
        return bool(self.data) and bool(self.errors)


# ──────────────────────────────────────────────
# GEODATAFRAME HELPERS
# ──────────────────────────────────────────────

def create_geodataframe(
    data: Union[gpd.GeoDataFrame, pd.DataFrame],
    params: dict,
) -> gpd.GeoDataFrame:
    """
    Standardise a GeoDataFrame or DataFrame into a GeoDataFrame with a location_id column.

    Args:
        data   : GeoDataFrame or DataFrame input.
        params : for DataFrame input, requires 'lon_name' and 'lat_name'.

    Returns:
        gpd.GeoDataFrame with 'location_id' column.
    """
    n_initial = len(data)

    if isinstance(data, gpd.GeoDataFrame):
        gdf = data.copy()
        gdf = gdf.drop_duplicates(subset=["geometry"])
        n_dropped = n_initial - len(gdf)
        print(f"Initial: {n_initial} | Dropped (geometry duplicates): {n_dropped} | Remaining: {len(gdf)}")
        pad_width = len(str(len(gdf)))
        gdf["location_id"] = [f"ID-{str(i + 1).zfill(max(pad_width, 3))}" for i in range(len(gdf))]

    elif isinstance(data, pd.DataFrame):
        lon_name = params.get("lon_name")
        lat_name = params.get("lat_name")
        if lon_name is None or lat_name is None:
            raise ValueError("params must include 'lon_name' and 'lat_name' for DataFrame input.")
        if lon_name not in data.columns or lat_name not in data.columns:
            raise ValueError(f"Columns '{lon_name}' and/or '{lat_name}' not found in DataFrame.")

        df = data.drop_duplicates(subset=[lon_name, lat_name]).copy()
        n_dropped = n_initial - len(df)
        print(f"Initial: {n_initial} | Dropped (coordinate duplicates): {n_dropped} | Remaining: {len(df)}")

        geometry = gpd.points_from_xy(df[lon_name], df[lat_name])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        gdf = gdf.dropna(subset=[lon_name, lat_name]).copy()
        gdf = gdf.sort_values(by=[lon_name, lat_name], ascending=[True, False]).reset_index(drop=True)
        pad_width = len(str(len(gdf)))
        gdf["location_id"] = [f"ID-{str(i + 1).zfill(max(pad_width, 3))}" for i in range(len(gdf))]

    else:
        raise TypeError("Input must be a GeoDataFrame or DataFrame.")

    return gdf


# ──────────────────────────────────────────────
# API ENDPOINTS
# ──────────────────────────────────────────────

def extract_aoi(
    gdf: gpd.GeoDataFrame = None,
    params_s: dict = None,
) -> tuple[dict, gpd.GeoDataFrame]:
    """
    Build a snapped BoundingBox dict and filtered GeoDataFrame from a GeoDataFrame or explicit bounds.

    Args:
        gdf      : source geometry; required unless override_gdf is set in params_s.
        params_s : spatial filters, explicit bounds (minx/miny/maxx/maxy), and/or 'override_gdf' flag.

    Returns:
        aoi     : BoundingBox dict snapped to 0.1 deg grid.
        gdf_aoi : filtered GeoDataFrame with 'location_id' column.
    """
    params_s   = params_s or {}
    bound_keys = {"minx", "miny", "maxx", "maxy"}
    override   = params_s.get("override_gdf", False)

    if bound_keys.issubset(params_s.keys()) and override:
        bounds = get_bounds(**{k: params_s[k] for k in bound_keys})
        polygon = box(bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"])
        gdf_aoi = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        gdf_aoi["location_id"] = "ID-000"

    else:
        if gdf is None:
            raise ValueError("gdf is required when override_gdf is not set in params_s.")

        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        gdf     = create_geodataframe(gdf, params_s)
        gdf_aoi = subset_geometry(gdf, params_s)

        if len(gdf_aoi) == 0:
            raise AssertionError("Geometry slice produced no elements. Check params_s filters.")

        minx, miny, maxx, maxy = gdf_aoi.total_bounds
        bounds = get_bounds(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    return BoundingBox(**bounds).to_dict(), gdf_aoi


def extract_data(
    params_request: dict,
    params_s: dict,
    params_t: Optional[dict] = None,
) -> DataRequestResult:
    """
    Download climate data for a given provider, field, AOI, and period.

    Args:
        params_request : required keys 'provider', 'field'; other keys override registered defaults.
        params_s       : BoundingBox dict (output of extract_aoi()).
        params_t       : keys 'start_date', 'end_date' (YYYY-MM-DD); optional 'date_mode' (bool, default False).

    Returns:
        DataRequestResult with .data keyed by year string (period) or YYYYMMDD_YYYYMMDD (range).
    """
    request = DataRequest.build(
        params_request=params_request,
        params_s=params_s,
        params_t=params_t,
    )
    print(request.describe())

    downloaded: Dict[str, xr.Dataset] = {}
    errors:     List[str] = []
    downloader = request.build_downloader()

    if request.date_mode:
        try:
            ds = downloader.download_range(request.start_date, request.end_date)
            downloaded[request.range_key] = ds
        except Exception as exc:
            msg = f"{request.provider} download_range error: {exc}"
            errors.append(msg)
            print(msg)

    else:
        years = request.years
        print(f"downloading {len(years)} year(s): {years[0]} to {years[-1]}")
        try:
            raw = downloader.download_period(years[0], years[-1])
            for year in years:
                ds = raw.get(str(year))
                if ds is not None:
                    downloaded[str(year)] = ds
                else:
                    errors.append(f"No data returned for year {year}")
        except Exception as exc:
            msg = f"{request.provider} download error: {exc}"
            errors.append(msg)
            print(msg)

    print(f"summary -> {len(downloaded)} downloaded | {len(errors)} failed")

    return DataRequestResult(
        request=request,
        data=downloaded,
        errors=errors,
    ).data