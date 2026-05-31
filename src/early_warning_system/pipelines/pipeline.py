from kedro.pipeline import Pipeline, node

from .utils import *
from .n00_create_target import *
from .n01_extract_data import *
from .n02_create_features import *
from .n03_create_mt import *
from .n04_hypertuning_model import *
from .n05_visualizations import *

def create_grid_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=get_spatial_params,
                inputs=['gdf_aoi', 'params:params_s'],
                outputs='dict_params_spatial',
                name='get_spatial_params',
                tags=['grid', 'target', 'grid_target', 'grid_predict'],
            ),
            node(
                func=create_grid,
                inputs=['dict_params_spatial'],
                outputs='ds_grid',
                name='create_grid',
                tags=['grid', 'grid_target', 'grid_predict'],
            ),
            node(
                func=create_geometry,
                inputs='ds_grid',
                outputs='gdf_grid',
                name='create_geometry',
                tags=['grid', 'grid_target'],
            ),
        ])

def create_features_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_features_topo,
                inputs=['ds_grid', 'ds_dem', 'params:topography'],
                outputs='ds_f_dem',
                name='create_features_topo',
                tags=['features'],
            ),
            node(
                func=create_features_forecast,
                inputs=['ds_fct_tmin', 'params:fct_tmin', 'params:params_clim_t'],
                outputs=['ds_f_fct_tmin', 'ds_f_clim_tmin'],
                name='create_features_fct_tmin',
                tags=['features'],
            ),
            node(
                func=create_features_forecast,
                inputs=['ds_fct_10u', 'params:fct_10u', 'params:params_clim_t'],
                outputs=['ds_f_fct_10u', 'ds_f_clim_10u'],
                name='create_features_fct_10u',
                tags=['features'],
            ),          
            node(
                func=create_features_forecast,
                inputs=['ds_fct_10v', 'params:fct_10v', 'params:params_clim_t'],
                outputs=['ds_f_fct_10v', 'ds_f_clim_10v'],
                name='create_features_fct_10v',
                tags=['features'],
            ),
            node(
                func=create_features_forecast,
                inputs=['ds_fct_msl', 'params:fct_msl', 'params:params_clim_t'],
                outputs=['ds_f_fct_msl', 'ds_f_clim_msl'],
                name='create_features_fct_msl',
                tags=['features'],
            ),
            node(
                func=create_features_forecast,
                inputs=['ds_fct_swc', 'params:fct_swc', 'params:params_clim_t'],
                outputs=['ds_f_fct_swc', 'ds_f_clim_swc'],
                name='create_features_fct_swc',
                tags=['features'],
            ),
            node(
                func=create_features_forecast,
                inputs=['ds_fct_tp', 'params:fct_tp', 'params:params_clim_t'],
                outputs=['ds_f_fct_tp', 'ds_f_clim_tp'],
                name='create_features_fct_tp',
                tags=['features'],
            ),
            node(
                func=create_features_forecast,
                inputs=['ds_fct_tp_chirps', 'params:fct_tp_chirps', 'params:params_clim_t'],
                outputs=['ds_f_fct_tp_chirps', 'ds_f_clim_tp_chirps'],
                name='create_features_fct_tp_chirps',
                tags=['features'],
            ),
        ])

def create_model_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=model_hypertuning,
                inputs=['df_mt_clean', 'params:params_t', 'params:prepare', 'params:hypertuning'],
                outputs=['df_tuning_preds', 'dict_tuning', 'pkl_tuning'],
                name='model_hypertuning',
                tags=['hypertuning']
            ),
            node(
                func=final_model,
                inputs=['df_mt_clean', 'params:params_t', 'dict_tuning', 'params:prepare', 'params:hypertuning'],
                outputs=['df_final_preds', 'dict_final', 'pkl_final'],
                name='final_model',
                tags=['final_model']
            )
        ])


def create_mt_target_coldspell_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_target,
                inputs=[ 
                    'df_tmin', 
                    'ds_aux_tmin',
                    'ds_grid',
                    'gdf_aoi', 
                    'params:params_t', 
                    'params:target_coldspell', 
                    'dict_params_spatial'
                ],
                outputs='df_target',
                name='create_target',
                tags=['target', 'grid_target'],
            ),
            node(
                func=create_mt_w_target,
                inputs=[
                    'params:mt', 
                    'params:params_t',
                    'ds_grid', 
                    'df_target', 
                    'ds_f_dem', 
                    'ds_f_fct_tmin',
                    'ds_f_fct_10u',
                    'ds_f_fct_10v',
                    'ds_f_fct_msl',
                    'ds_f_fct_swc',
                ],
                outputs='df_mt',
                name='create_raw_mt',
                tags=['mt'],
            ),
            node(
                func=clean_mt,
                inputs=['params:mt', 'df_mt'],
                outputs='df_mt_clean',
                name='clean_mt',
                tags=['mt']
            ),
        ])

def create_mt_target_rainfall_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_target,
                inputs=[ 
                    'df_prcp', 
                    'ds_aux_prcp',
                    'ds_grid',
                    'gdf_aoi', 
                    'params:params_t', 
                    'params:target_rainfall', 
                    'dict_params_spatial'
                ],
                outputs='df_target',
                name='create_target',
                tags=['target', 'grid_target'],
            ),
            node(
                func=create_mt_w_target,
                inputs=[
                    'params:mt', 
                    'params:params_t',
                    'ds_grid', 
                    'df_target', 
                    'ds_f_dem', 
                    'ds_f_fct_tmin',
                    'ds_f_fct_10u',
                    'ds_f_fct_10v',
                    'ds_f_fct_msl',
                    'ds_f_fct_swc',
                    'ds_f_fct_tp',
                    'ds_f_fct_tp_chirps',
                ],
                outputs='df_mt',
                name='create_raw_mt',
                tags=['mt'],
            ),
            node(
                func=clean_mt,
                inputs=['params:mt', 'df_mt'],
                outputs='df_mt_clean',
                name='clean_mt',
                tags=['mt']
            ),
        ])

def create_visualization_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=plot_metrics_classif_by_group,
                inputs=['df_final_preds', 'params:plot_metrics'],
                outputs='plt_metrics',
                name='plot_metrics',
                tags=['viz']
            ),
            node(
                func=plot_metric_scatter_geo,
                inputs=['df_final_preds', 'gdf_aoi', 'params:plot_scatter'],
                outputs='plt_scatter',
                name='plot_scatter',
                tags=['viz']
            ),
            node(
                func=plot_precision_recall_vs_threshold,
                inputs=['df_final_preds', 'params:plot_curves'],
                outputs='plt_curves',
                name='plot_curves',
                tags=['viz']
            ),
        ])


def create_coldspells_pipeline(**kwargs) -> Pipeline:

    grid = create_grid_pipeline()
    features = create_features_pipeline()
    target_mt = create_mt_target_coldspell_pipeline()
    model = create_model_pipeline()
    viz = create_visualization_pipeline()
    return grid + features + target_mt + model + viz

def create_rainfall_pipeline(**kwargs) -> Pipeline:

    grid = create_grid_pipeline()
    features = create_features_pipeline()
    target_mt = create_mt_target_rainfall_pipeline()
    model = create_model_pipeline()
    viz = create_visualization_pipeline()
    return grid + features + target_mt + model + viz


# DOWNLOAD DATA

def update_download_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=extract_data,
                inputs=['params:request_tmin', 'dict_params_spatial', 'params:params_t'],
                outputs='ds_d_fct_tmin',
                name='download_features_tmin',
                tags=['features'],
            ),
            node(
                func=extract_data,
                inputs=['params:request_10u', 'dict_params_spatial', 'params:params_t'],
                outputs='ds_d_fct_10u',
                name='download_features_10u',
                tags=['features'],
            ),
            node(
                func=extract_data,
                inputs=['params:request_10v', 'dict_params_spatial', 'params:params_t'],
                outputs='ds_d_fct_10v',
                name='download_features_10v',
                tags=['features'],
            ),
            node(
                func=extract_data,
                inputs=['params:request_msl', 'dict_params_spatial', 'params:params_t'],
                outputs='ds_d_fct_msl',
                name='download_features_msl',
                tags=['features'],
            ),
            node(
                func=extract_data,
                inputs=['params:request_swc', 'dict_params_spatial', 'params:params_t'],
                outputs='ds_d_fct_swc',
                name='download_features_swc',
                tags=['features'],
            ),
            node(
                func=extract_data,
                inputs=['params:request_tp', 'dict_params_spatial', 'params:params_t'],
                outputs='ds_d_fct_tp',
                name='download_features_tp',
                tags=['features'],
            ),
            node(
                func=extract_data,
                inputs=['params:request_tp_chirps', 'dict_params_spatial', 'params:params_t'],
                outputs='ds_d_fct_tp_chirps',
                name='download_features_tp_chirps',
                tags=['features'],
            ),
        ])

def update_features_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_features_fct_clim,
                inputs=['ds_d_fct_tmin', 'ds_f_clim_tmin', 'params:fct_tmin', 'params:params_t'],
                outputs='ds_u_fct_tmin',
                name='create_features_fct_tmin',
                tags=['features'],
            ),
            node(
                func=create_features_fct_clim,
                inputs=['ds_d_fct_10u', 'ds_f_clim_10u', 'params:fct_10u', 'params:params_t'],
                outputs='ds_u_fct_10u',
                name='create_features_fct_10u',
                tags=['features'],
            ),
            node(
                func=create_features_fct_clim,
                inputs=['ds_d_fct_10v', 'ds_f_clim_10v', 'params:fct_10v', 'params:params_t'],
                outputs='ds_u_fct_10v',
                name='create_features_fct_10v',
                tags=['features'],
            ),
            node(
                func=create_features_fct_clim,
                inputs=['ds_d_fct_msl', 'ds_f_clim_msl', 'params:fct_msl', 'params:params_t'],
                outputs='ds_u_fct_msl',
                name='create_features_fct_msl',
                tags=['features'],
            ),
            node(
                func=create_features_fct_clim,
                inputs=['ds_d_fct_swc', 'ds_f_clim_swc', 'params:fct_swc', 'params:params_t'],
                outputs='ds_u_fct_swc',
                name='create_features_fct_swc',
                tags=['features'],
            ),
            node(
                func=create_features_fct_clim,
                inputs=['ds_d_fct_tp', 'ds_f_clim_tp', 'params:fct_tp', 'params:params_t'],
                outputs='ds_u_fct_tp',
                name='create_features_fct_tp',
                tags=['features'],
            ),
            node(
                func=create_features_fct_clim,
                inputs=['ds_d_fct_tp_chirps', 'ds_f_clim_tp_chirps', 'params:fct_tp_chirps', 'params:params_t'],
                outputs='ds_u_fct_tp_chirps',
                name='create_features_fct_tp_chirps',
                tags=['features'],
            ),
        ])


def update_mt_predict_coldspell_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_mt_predict,
                inputs=[
                    'params:mt', 
                    'params:params_t',
                    'ds_grid',  
                    'ds_f_dem', 
                    'ds_u_fct_tmin',
                    'ds_u_fct_10u',
                    'ds_u_fct_10v',
                    'ds_u_fct_msl',
                    'ds_u_fct_swc',
                ],
                outputs='df_mt',
                name='create_raw_mt',
                tags=['mt'],
            ),
            node(
                func=clean_mt,
                inputs=['params:mt', 'df_mt'],
                outputs='df_mt_clean_predict',
                name='clean_mt',
                tags=['mt']
            ),
        ])

def update_mt_predict_rainfall_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_mt_predict,
                inputs=[
                    'params:mt', 
                    'params:params_t',
                    'ds_grid',  
                    'ds_f_dem', 
                    'ds_u_fct_tmin',
                    'ds_u_fct_10u',
                    'ds_u_fct_10v',
                    'ds_u_fct_msl',
                    'ds_u_fct_swc',
                    'ds_u_fct_tp',
                    'ds_u_fct_tp_chirps',
                ],
                outputs='df_mt',
                name='create_raw_mt',
                tags=['mt'],
            ),
            node(
                func=clean_mt,
                inputs=['params:mt', 'df_mt'],
                outputs='df_mt_clean_predict',
                name='clean_mt',
                tags=['mt']
            ),
        ])

def update_predict_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=predict_target,
                inputs=['df_mt_clean_predict', 'pkl_final', 'params:prepare', 'params:predict'],
                outputs='df_update_preds',
                name='predict_target',
                tags=['predict']
            ),
            node(
                func=build_alerts_with_admin,
                inputs=['df_update_preds', 'gdf_hierarchy'],
                outputs='df_update_alerts',
                name='build_alerts',
                tags=['predict']
            ),
        ])

def update_coldspell_pipeline(**kwargs) -> Pipeline:

    grid = create_grid_pipeline()
    download = update_download_pipeline()
    features = update_features_pipeline()
    mt = update_mt_predict_coldspell_pipeline()
    predict = update_predict_pipeline()
    return grid + download + features + mt + predict

def update_rainfall_pipeline(**kwargs) -> Pipeline:

    grid = create_grid_pipeline()
    download = update_download_pipeline()
    features = update_features_pipeline()
    mt = update_mt_predict_rainfall_pipeline()
    predict = update_predict_pipeline()
    return grid + download + features + mt + predict