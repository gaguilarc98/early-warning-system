from datetime import datetime, timedelta
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path

PROJECT_PATH = Path(__file__).parents[1]  # points to early-warning-system/
bootstrap_project(PROJECT_PATH)

date_fct = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d') 


runtime_params_tmin = {
    'country': 'bolivia', # Local folder name to store data 
    'provider': 'ECMWF', # Data provider, can be 'ERA5' or 'UCSB'
    'field': 'tmin', # Short variable name, can be 'swc', 'prcp', 'tmin', 'tmax'
    'peril': 'coldspell',
    'start_date': date_fct,
    'end_date': date_fct,
}
runtime_params_prcp = runtime_params_tmin.copy()
runtime_params_prcp['field'] = 'prcp'
runtime_params_prcp['peril'] = 'rainfall'

def run_download_nodes():
    with KedroSession.create(project_path=Path(PROJECT_PATH), runtime_params = runtime_params_tmin) as session:
        session.run(
            pipeline_names = ['update_coldspell'],
            node_names = [
                'get_spatial_params', 'download_features_tmin', 'download_features_10u', 
                'download_features_10v', 'download_features_msl', 'download_features_swc', 
                'download_features_tp', 'download_features_tp_chirps',
            ]
        )
    
def run_feature_nodes():
    with KedroSession.create(project_path=Path(PROJECT_PATH), runtime_params = runtime_params_tmin) as session:
        session.run(
            pipeline_names = ['update_coldspell'],
            node_names = [
                'create_features_fct_tmin', 'create_features_fct_10u', 
                'create_features_fct_10v', 'create_features_fct_msl', 
                'create_features_fct_swc', 'create_features_fct_tp', 
                'create_features_fct_tp_chirps'
            ]
        )

def run_fct_nodes():
    with KedroSession.create(project_path=Path(PROJECT_PATH), runtime_params = runtime_params_tmin) as session:
        session.run(
            pipeline_names = ['update_coldspell'],
            tags=['mt', 'predict']
        )
    
    with KedroSession.create(project_path=Path(PROJECT_PATH), runtime_params = runtime_params_prcp) as session:
        session.run(
            pipeline_names = ['update_rainfall'],
            tags=['mt', 'predict']
        )
    

def kedro_run_full():
    # Download files
    try:
        run_download_nodes()
    except Exception as e:
        print('An error ocurred when trying to download files from source.')
        print(e)
    # Compute features with new data
    try:
        run_feature_nodes()
    except Exception as e:
        print('An error ocurred when trying to create features.')
        print(e)
    # Build MT and predict with new data
    try:
        run_fct_nodes()
    except Exception as e:
        print('An error ocurred when trying to predict events.')
        print(e)

#kedro_run_full()

if __name__ == "__main__":
    kedro_run_full()