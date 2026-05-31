from .utils import *

from .a04_score_helpers import *

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, LeaveOneGroupOut, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelEncoder, label_binarize

import optuna

from deltalake import DeltaTable


#——————————————————————————————————————
# SPLIT FUNCTIONS
#——————————————————————————————————————


def backtest_split(
    df,
    split_col: str,
    n_slices: int=0,
    frac_slices: float=0.2,
):
    if n_slices>0:
        # Get the number of unique slices and sort them in descending order
        slices = np.flip(np.sort(df[split_col].unique()))
        if n_slices>len(slices):
            raise ValueError('n_slices must be an integer less than the number of slices.')
        backtest_slices = slices[:n_slices] 
        # Extract the selected number of slices for backtest and keep the rest for training
        df_back = df[df[split_col].isin(backtest_slices)].copy() # Select backtest slices
        df_train = df[~df[split_col].isin(backtest_slices)].copy() # Select backtest slices
    elif frac_slices>0 and frac_slices<1:
        # Get the sizes of slices and calculate its cumulative distribution in descending order
        sizes = df[split_col].value_counts(normalize=True).sort_index(ascending=False)
        cum_sizes = sizes.cumsum()
        backtest_slices = cum_sizes[cum_sizes<frac_slices].index
        # Extract the selected proportion of slices for backtest and keep the rest for training
        df_back = df[df[split_col].isin(backtest_slices)].copy()
        df_train = df[~df[split_col].isin(backtest_slices)].copy()
    else:
        raise ValueError('frac_slices must be a float between 0 and 1.')
    return df_train, df_back


def get_x_y(df, ignore_cols, target):
    '''Returns the spliting into features (X), descriptive variables for target (y_full) and target(y)'''
    X = df[[col for col in df.columns if col not in ignore_cols]].copy()
    y_full = df[ignore_cols].copy()
    y = df[target].copy()
    return X, y_full, y


#——————————————————————————————————————
# MODEL SELECTOR AND CONSTRUCTOR
#——————————————————————————————————————


def select_model(algorithm, n_classes, categorical_cols=None):
    # Modeling and training with the algorithm and the values of the hyperparameters to perform the tuning
    if algorithm == 'hgbr':
        mod = HistGradientBoostingRegressor(
            random_state=24,
            verbose=0,
            categorical_features=categorical_cols
        )
        fit_params = {}
    elif algorithm == 'lgbr':
        mod = LGBMRegressor(
            random_state=24,
            verbose=0,
            importance_type='gain'
        )
        fit_params = {"categorical_feature":categorical_cols}
    elif algorithm == 'xgbr':
        mod = XGBRegressor(
            random_state=24,
            tree_method='hist',
            enable_categorical=True
        )
        fit_params = {}
    elif algorithm == 'hgbc':
        mod = HistGradientBoostingClassifier(
            random_state=24,
            verbose=0,
            class_weight='balanced',
            categorical_features=categorical_cols
        )
        fit_params = {}
    elif algorithm == 'xgbc':
        mod = XGBClassifier(
            tree_method='hist',
            objective='multi:softmax' if n_classes > 2 else 'binary:logistic',
            n_jobs=1,
            #num_class=n_classes,
            enable_categorical=True,
        )
        fit_params = {}
    elif algorithm == 'lgbc':
        mod = LGBMClassifier(
            verbose=0,
            n_jobs=1,
            importance_type='gain',
            objective='multiclass' if n_classes > 2 else 'binary',
            #num_class=n_classes,
        )
        fit_params = {"categorical_feature":categorical_cols}

    else:
        raise ValueError(f'No support for {algorithm} method.')
    
    return mod, fit_params


#———————————————————————————————————————————
# OBJECTIVE FUNCTIONS
#———————————————————————————————————————————


def resolve_scorer(scorer):
    """
    Accept:
    - string (built-in scorer name)
    - sklearn scorer object (from make_scorer)
    - callable(estimator, X, y)
    """

    if isinstance(scorer, str):
        return get_scorer(scorer)

    if callable(scorer):
        return scorer

    raise ValueError("Invalid scorer provided.")

class CVObjective:

    def __init__(
        self,
        mod,
        param_config,
        X,
        y,
        scorer,
        cv,
        groups=None,
        fit_params={},
    ):
        self.mod = mod
        self.param_config = param_config
        self.X = X
        self.y = y
        self.scorer = resolve_scorer(scorer)
        self.cv = cv
        self.groups = {'groups': groups}
        self.fit_params = fit_params

    def suggest_from_config(self, trial, name, spec):

        if 'type' not in spec or spec['type'] not in ['float', 'int', 'categorical']:
            raise KeyError(f'The specifications did not specified a valid type for {name}')
        
        if spec['type'] == 'float':
            return trial.suggest_float(
                name,
                spec['low'],
                spec['high'],
                step=spec.get('step', None),
                log=spec.get('log', False)
            )

        if spec['type'] == 'int':
            return trial.suggest_int(
                name,
                spec['low'],
                spec['high'],
                step=spec.get('step', 1),
                log=spec.get('log', False)
            )

        if spec['type'] == 'categorical':
            return trial.suggest_categorical(
                name,
                spec['choices']
            )

    def __call__(self, trial):

        params = {
            name: self.suggest_from_config(trial, name, spec)
            for name, spec in self.param_config.items()
        }

        train_scores = []
        val_scores = []

        for fold_id, (train_idx, val_idx) in enumerate(
            self.cv.split(self.X, self.y, **self.groups)
        ):

            X_train = self.X.iloc[train_idx]
            X_val   = self.X.iloc[val_idx]

            y_train = self.y.iloc[train_idx]
            y_val   = self.y.iloc[val_idx]

            model = clone(self.mod)
            model.set_params(**params)
            model.fit(X_train, y_train, **self.fit_params)

            train_score = self.scorer(model, X_train, y_train)
            val_score = self.scorer(model, X_val, y_val)

            train_scores.append(train_score)
            val_scores.append(val_score)

            # Optional pruning
            trial.report(np.mean(val_scores), step=fold_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        mean_train = np.mean(train_scores)
        mean_val   = np.mean(val_scores)

        # Store diagnostics
        trial.set_user_attr("mean_train", mean_train)
        trial.set_user_attr("gap_train_test", mean_train - mean_val)

        return mean_val


#————————————————————————————————————————————
# HYPERTUNING FUNCTION
#————————————————————————————————————————————


class DataPreparer:
    def __init__(self, params_prepare: dict):
        self.params = params_prepare
        
        # Will be filled after prepare()
        self.X_train = None
        self.y_full_train = None
        self.y_train = None

        self.X_back = None
        self.y_full_back = None
        self.y_back = None

        self.X_out = None
        self.y_full_out = None
        self.y_out = None

        self.categorical_cols = None
    
    @property
    def train(self):
        return self.X_train, self.y_full_train, self.y_train

    @property
    def back(self):
        return self.X_back, self.y_full_back, self.y_back

    @property
    def out(self):
        return self.X_out, self.y_full_out, self.y_out

    def prepare(self, df_mt_orig: pd.DataFrame):
        df = df_mt_orig.copy()

        ignore_cols = self.params['ignore_cols']
        target = self.params['target']
        n_months_backtest = self.params['n_months_backtest']
        subset_fraq = self.params.get('subset_fraq', 0.25)
        subset_col = self.params.get('subset_by', None)
        stratify_col = self.params.get('stratify_by', None)
        remove_nans_from = self.params.get('remove_nans_from', [])

        # Remove NaNs
        df = df.dropna(subset=remove_nans_from)

        # Detect categorical cols
        self.categorical_cols = list(
            set(
                df.select_dtypes(include=['object', 'category', 'bool']).columns
            ) - set(ignore_cols)
        )

        for col in self.categorical_cols:
            df[col] = df[col].astype('category')

        # Optional Out-of-sample split
        if subset_col is not None:

            if stratify_col is None:
                df_aux = df[[subset_col]].drop_duplicates()
                stratify_values = None
            else:
                df_aux = df[[subset_col, stratify_col]].drop_duplicates()
                stratify_values = df_aux[stratify_col]

            split_values = df_aux[subset_col]

            train_split, test_split = train_test_split(
                split_values,
                test_size=subset_fraq,
                random_state=42,
                stratify=stratify_values
            )

            df_out = df[df[subset_col].isin(test_split)].copy()
            df_in = df[df[subset_col].isin(train_split)].copy()

            self.X_out, self.y_full_out, self.y_out = get_x_y(
                df_out, ignore_cols, target
            )

        else:
            df_in = df
            self.X_out = self.y_full_out = self.y_out = None

        # Backtest split
        df_train, df_back = backtest_split(
            df_in,
            split_col='yearmon',
            n_slices=n_months_backtest
        )

        self.X_train, self.y_full_train, self.y_train = get_x_y(
            df_train, ignore_cols, target
        )

        self.X_back, self.y_full_back, self.y_back = get_x_y(
            df_back, ignore_cols, target
        )

        return self


def sort_models(
    df_results,
):
    df = df_results.copy()
    df['rank_dif'] = df['gap_train_test'].rank(method='dense', ascending=True, na_option='bottom')
    #df['rank_train'] = df['mean_train_score'].rank(method='dense', ascending=False, na_option='bottom')
    df['rank_test'] = df['mean_test_score'].rank(method='dense', ascending=False, na_option='bottom')
    df['mean_rank'] = np.apply_along_axis(np.mean, 1, df[['rank_dif', 'rank_test']])

    df = df.reset_index(drop=True)
    df = df.sort_values(by=['mean_rank', 'mean_test_score'], ascending=[True, True]).reset_index(drop=True)
    return df


def add_model_prob_columns(y, y_prob, n_classes, subset):
    y_full = y.copy()
    for i in range(n_classes):
        y_full[f'prob_class_{int(i)}'] = y_prob[:,i]
    
    y_full['subset'] = subset
    return y_full

def add_model_pred_columns(y, y_pred, subset, n_classes=None):
    y_full = y.copy()
    if n_classes is None:
        y_full[f'pred_target'] = y_pred
    else:
        for i in range(n_classes):
            y_full[f'prob_class_{int(i)}'] = y_pred[:,i]
    
    y_full['subset'] = subset
    return y_full


def filter_delta(
    dt: DeltaTable,
    params_t: dict,
) -> pd.DataFrame:
    """
    Materialize a DeltaTable to pd.DataFrame filtered to the requested date range.
    Filters on the 'yearmon' partition column for partition pruning, then trims
    the edges with an exact date filter on 'time'.
    Args:
        - dt       : (DeltaTable) Loaded from catalog, no data in RAM yet
        - params_t : (dict) Temporal parameters, must contain 'start_date' and 'end_date'
    Returns:
        pd.DataFrame filtered to the requested date range
    """
    start = pd.to_datetime(params_t['start_date'])
    end   = pd.to_datetime(params_t['end_date'])

    # Generate all 'YYYY-MM' strings covering the requested range
    yearmons = (
        pd.date_range(start=start, end=end, freq='MS')
        .strftime('%Y%m')
        .tolist()
    )

    # Partition pruning happens here -- only matching yearmon folders are read
    df = dt.to_pandas(filters=[('yearmon', 'in', yearmons)])

    # Trim edges -- boundary months may contain dates outside the range
    df = df.query("time >= @start and time <= @end").reset_index(drop=True)
    return df


def model_hypertuning(
    df_mt_clean: pd.DataFrame,
    params_t: dict,
    params_prepare_orig: dict,
    params_hypertuning_orig: dict,
) -> tuple:
    """Perform grid search cross-validation for machine learning models.
    Args:
        - df_mt_orig : (pd.DataFrame) Original Master Table with features and target variable.
        - params_hypertuning : (dict) Dictionary with parameters to test during hypertuning.
    Returns:
        tuple: A tuple containing the following elements:
            - df_predictions : (pd.DataFrame) Dataframe with the predictions from best model
            - dict_results : (Dictionary) Dictionary containing results, importances and metrics
            - mod_best : (pickle) Fitted model to ve saved in binary format
    """

    peril = params_prepare_orig['peril']
    params_prepare = params_prepare_orig[peril]
    params_hypertuning = params_hypertuning_orig[peril]

    df_mt_orig = filter_delta(df_mt_clean, params_t)

    preparer = DataPreparer(params_prepare)
    preparer.prepare(df_mt_orig)
    
    categorical_cols = preparer.categorical_cols

    X_train, y_full_train, y_train = preparer.train
    X_back, y_full_back, y_back = preparer.back
    X_out, y_full_out, y_out = preparer.out
    
    """
    ignore_cols = params_hypertuning['ignore_cols']
    target = params_hypertuning['target']
    station_fraq = params_hypertuning['station_fraq']
    n_months_backtest = params_hypertuning['n_months_backtest']

    remove_nans_from = params_hypertuning['remove_nans_from']
    df_mt_orig = df_mt_orig.dropna(subset=remove_nans_from).copy()

    # Transform categorical variables
    categorical_cols = list(set(df_mt_orig.select_dtypes(include=['object', 'category', 'bool']).columns)-set(ignore_cols))
    for col in categorical_cols:
        df_mt_orig[col] = df_mt_orig[col].astype('category')
    
    # Split stations to test model out-of-sample
    df_aux = df_mt_orig[['decadeCode', 'cluster']].drop_duplicates()
    list_stations = df_aux['decadeCode']
    cluster_stations = df_aux['cluster']
    stations_train, stations_test = train_test_split(list_stations, test_size=station_fraq, random_state=42, stratify=cluster_stations)

    # Split in-sample data into train, backtest and out-of-sample
    df_out = df_mt_orig[df_mt_orig['decadeCode'].isin(stations_test)].copy()
    df_in = df_mt_orig[df_mt_orig['decadeCode'].isin(stations_train)].copy()
    df_train, df_back = backtest_split(df_in, split_col='yearmon', n_slices=n_months_backtest)
    
    # Combine X_train and X_val
    X_train, y_full_train, y_train = get_x_y(df_train, ignore_cols, target)
    X_back, y_full_back, y_back = get_x_y(df_back, ignore_cols, target)
    X_out, y_full_out, y_out = get_x_y(df_out, ignore_cols, target)
    """
    n_trials = params_hypertuning['n_trials']
    target = params_prepare['target']
    n_classes = len(df_mt_orig[target].unique())

    algorithm = params_hypertuning['algorithm']
    grid_values = params_hypertuning['grid_values'][algorithm]

    # Define groups for cross-validation
    groups = y_full_train['yearmon']
    gss = GroupShuffleSplit(n_splits=4, test_size=0.25, random_state=42)

    # Scoring function
    if algorithm in ['hgbc', 'xgbc', 'lgbc']:
        metric = "roc_auc_ovr" if n_classes > 2 else 'roc_auc'
    elif algorithm in ['hgbr', 'xgbr', 'lgbr']:
        metric = 'r2'

    # Select model
    mod, fit_params = select_model(algorithm, n_classes, categorical_cols)

    """
    # Modeling and training with the algorithm and the values of the hyperparameters to perform the tuning
    grid_cv = GridSearchCV(
        mod,
        param_grid=grid_values,
        scoring="roc_auc_ovr" if n_classes > 2 else 'roc_auc',#'neg_root_mean_squared_error',
        cv=gss,
        return_train_score=True,
        verbose=1,
        n_jobs=-1
    )
    """

    '''
    grid_cv = RandomizedSearchCV(
        mod,
        param_distributions = grid_values,
        scoring = 'r2',
        n_iter = 50,
        cv = gss,
        error_score='raise',
        return_train_score = True,
        verbose = 1,
        n_jobs = -1,
        random_state = 42
    )
    '''
    #grid_cv.fit(X_train, y_train, groups = groups, **fit_params)
    objective = CVObjective(
        mod,
        param_config=grid_values,
        X=X_train,
        y=y_train,
        scorer=metric, #gini_scorer
        cv=gss,
        groups=groups,
        fit_params=fit_params
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=4)  

    # Show the results of the different iterations carried out
    df_tuning_results = []
    for t in study.trials:
        df_tuning_results.append({
            "trial_number": t.number,
            "params": t.params,   # <-- dictionary stored as-is
            "state": t.state.name,
            "mean_test_score": t.value,
            "mean_train_score": t.user_attrs.get("mean_train"),
            "gap_train_test": t.user_attrs.get("gap_train_test"),
        })
    df_tuning_results = pd.DataFrame(df_tuning_results)

    # Add the following columns: rundate, algorithm
    df_tuning_results.insert(0, "date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df_tuning_results.insert(1, "algorithm", algorithm)

    # Sort the models with a custom criteria and select the highest-ranked model
    df_tuning_results = sort_models(df_tuning_results)
    best_params = df_tuning_results.loc[0, 'params']

    # Create a new instance of the model with the best parameters and fit the model
    mod_best = clone(mod)
    mod_best.set_params(**best_params)
    mod_best.fit(X_train, y_train)

    # Use the best model to make predictions for the training, backtesting and out-of-sample datasets
    preds_y_train = mod_best.predict(X_train)
    preds_y_back = mod_best.predict(X_back)
    preds_y_out = mod_best.predict(X_out)

    # Get metrics and predictions DataFrames
    df_metrics = []
    if algorithm in ['hgbc', 'xgbc', 'lgbc']:
        #TODO: put predictions based on if it is a classification or a regression model
        probs_y_train = mod_best.predict_proba(X_train)#[:,1]
        probs_y_back = mod_best.predict_proba(X_back)#[:,1]
        probs_y_out = mod_best.predict_proba(X_out)#[:,1]
        y_full_train = add_model_pred_columns(y_full_train, probs_y_train, 'train', n_classes)
        y_full_back = add_model_pred_columns(y_full_back, probs_y_back, 'backtest', n_classes)
        y_full_out = add_model_pred_columns(y_full_out, probs_y_out, 'out_of_sample', n_classes)

        df_predictions = pd.concat([y_full_train, y_full_back, y_full_out], axis=0, ignore_index= True)

        df_metrics.append(get_metrics_multiclassif(y_train, probs_y_train, preds_y_train, mod_best, name_set='train'))
        df_metrics.append(get_metrics_multiclassif(y_back, probs_y_back, preds_y_back, mod_best, name_set='backtest'))
        df_metrics.append(get_metrics_multiclassif(y_out, probs_y_out, preds_y_out, mod_best, name_set='out_of_sample'))
    elif algorithm in ['hgbr', 'xgbr', 'lgbr']:
        y_full_train = add_model_pred_columns(y_full_train, preds_y_train, 'train')
        y_full_back = add_model_pred_columns(y_full_back, preds_y_back, 'backtest')
        y_full_out = add_model_pred_columns(y_full_out, preds_y_out, 'out_of_sample')

        df_predictions = pd.concat([y_full_train, y_full_back, y_full_out], axis=0, ignore_index= True)

        df_metrics.append(get_metrics_reg(y_train, preds=preds_y_train, name_set='train'))
        df_metrics.append(get_metrics_reg(y_back, preds=preds_y_back, name_set='backtest'))
        df_metrics.append(get_metrics_reg(y_out, preds=preds_y_out, name_set='out_of_sample'))

    df_metrics = pd.concat(df_metrics, axis=1).reset_index(names='metric')

    # Get importance DataFrame
    df_importances = get_feature_importance(mod_best, X_train, y_train, params_hypertuning)
    df_importances = df_importances.to_frame()

    dict_results = {
        'tuning_results': df_tuning_results,
        'metrics': df_metrics,
        'importance': df_importances
    }
    
    return (
        df_predictions,
        dict_results,
        mod_best
    )


#————————————————————————————————————————
# FINAL MODEL TRAINING
#————————————————————————————————————————


def final_model(
    df_mt_clean: pd.DataFrame, 
    params_t: dict,
    dict_results: dict, 
    params_prepare_orig: dict,
    params_hypertuning_orig: dict=None
) -> tuple:
    """
    Train and calibrate the final model with the best hyperparameters from hypertuning
    Args:
        - df_mt_raw : (pd.DataFrame) DataFrame with original features and target to train on.
        - dict_results : (dict) Dictionary containing the hypertuning results.
        - params (dict): Dictionary containing model parameters.
    Returns:
        tuple : A tuple containing the following:
            - df_final_preds : DataFrame containing independent variables with its dimensions and predictions.
            - dict_summary : Dict object containing metrics and variable importances of the final model.
            - mod_isotonic: Calibrated model.

    Notes:
        - This function performs modeling and evaluation of the final model using the provided parameters.
        - It calculates various metrics such as AUC, Gini, and KS for training, validation, and backtest datasets.
        - It generates decile-wise summary statistics for the backtest dataset.
    """

    peril = params_prepare_orig['peril']
    params_prepare = params_prepare_orig[peril]
    params_hypertuning = params_hypertuning_orig[peril]


    df_mt_orig = filter_delta(df_mt_clean, params_t)
    preparer = DataPreparer(params_prepare)
    preparer.prepare(df_mt_orig)

    categorical_cols = preparer.categorical_cols
    
    X_train, y_full_train, y_train = preparer.train
    X_back, y_full_back, y_back = preparer.back
    X_out, y_full_out, y_out = preparer.out

    X_cal = pd.concat([X_back, X_out])
    y_cal = pd.concat([y_back, y_out])

    # Additional information from parameters
    subset_fraq = params_prepare['subset_fraq']
    n_months_backtest = params_prepare['n_months_backtest']
    target = params_prepare['target']
    n_classes = len(df_mt_orig[target].unique())

    num_quantiles = params_prepare["num_quantiles"]
    algorithm = params_hypertuning['algorithm']
    index_best = int(params_hypertuning['index_best'])
    params_algorithm = dict_results['tuning_results'].loc[index_best, 'params']

    """
    # Modeling and training with the algorithm and the values of the hyperparameters to perform the tuning
    if algorithm == 'hgbr':
        mod = HistGradientBoostingRegressor(
            verbose=0,
            categorical_features=categorical_cols,
            **params_algorithm
        )
        fit_params = {}
    elif algorithm == 'lgbr':
        mod = LGBMRegressor(
            verbose=0,
            importance_type='gain',
            **params_algorithm
        )
        fit_params = {"categorical_feature":categorical_cols}
    elif algorithm == 'xgbr':
        mod = XGBRegressor(
            tree_method='hist',
            enable_categorical=True,
            **params_algorithm
        )
        fit_params = {}
    elif algorithm == 'hgbc':
        mod = HistGradientBoostingClassifier(
            verbose=0,
            class_weight='balanced',
            categorical_features=categorical_cols,
            **params_algorithm
        ) 
        fit_params = {}
    elif algorithm == 'xgbc':
        mod = XGBClassifier(
            tree_method='hist',
            objective='multi:softmax' if n_classes > 2 else 'binary:logistic',
            n_jobs=1,
            #num_class=n_classes,
            enable_categorical=True,
            **params_algorithm
        )
        fit_params = {}
    elif algorithm == 'lgbc':
        mod = LGBMClassifier(
            verbose=0,
            n_jobs=1,
            importance_type='gain',
            objective='multiclass' if n_classes > 2 else 'binary',
            #num_class=n_classes,
            **params_algorithm
        )
        fit_params = {"categorical_feature":categorical_cols}
    #elif algorithm == "CatBoostClassifier":
    #    mod = CatBoostClassifier(random_state=24, 
    #                             eval_metric='AUC')
    #    fit_params = {"cat_features":categorical_cols}
    else:
        raise ValueError(f'No support for {algorithm} method.')
    """

    # Select and train final model
    mod, fit_params = select_model(algorithm, n_classes, categorical_cols)
    mod.set_params(**params_algorithm)
    mod.fit(X_train, y_train, **fit_params)
    
    # Calibrate the model with the calibration set
    mod_isotonic = CalibratedClassifierCV(mod, method="isotonic", cv="prefit")
    mod_isotonic.fit(X_cal, y_cal)

    preds_y_train = mod_isotonic.predict(X_train)
    preds_y_back = mod_isotonic.predict(X_back)
    preds_y_out = mod_isotonic.predict(X_out)

    # Get predictions and metrics DataFrames
    df_metrics = []
    if algorithm in ['hgbc', 'xgbc', 'lgbc']:
        # Get the predicted values
        probs_y_train = mod_isotonic.predict_proba(X_train)#[:,1]
        probs_y_back = mod_isotonic.predict_proba(X_back)#[:,1]
        probs_y_out = mod_isotonic.predict_proba(X_out)#[:,1]

        y_full_train = add_model_pred_columns(y_full_train, probs_y_train, 'train', n_classes)
        y_full_back = add_model_pred_columns(y_full_back, probs_y_back, 'backtest', n_classes)
        y_full_out = add_model_pred_columns(y_full_out, probs_y_out, 'out_of_sample', n_classes)

        df_final_preds = pd.concat([y_full_train, y_full_back, y_full_out], axis=0, ignore_index= True)
        for c in mod_isotonic.classes_:
            df_final_preds[f'quantile_class_{int(c)}'] = df_final_preds.groupby(
                ['idStation']
            )[f'prob_class_{int(c)}'].transform(
                lambda x: 1 + pd.qcut(x.rank(method="first", ascending=False), num_quantiles, labels=False)
            )

        df_metrics.append(get_metrics_multiclassif(y_train, probs_y_train, preds_y_train, mod_isotonic, name_set='train'))
        df_metrics.append(get_metrics_multiclassif(y_back, probs_y_back, preds_y_back, mod_isotonic, name_set='backtest'))
        df_metrics.append(get_metrics_multiclassif(y_out, probs_y_out, preds_y_out, mod_isotonic, name_set='out_of_sample'))
    elif algorithm in ['hgbr', 'xgbr', 'lgbr']:
        y_full_train = add_model_pred_columns(y_full_train, preds_y_train, 'train')
        y_full_back = add_model_pred_columns(y_full_back, preds_y_back, 'backtest')
        y_full_out = add_model_pred_columns(y_full_out, preds_y_out, 'out_of_sample')

        df_final_preds = pd.concat([y_full_train, y_full_back, y_full_out], axis=0, ignore_index= True)

        df_metrics.append(get_metrics_reg(y_train, preds=preds_y_train, name_set='train'))
        df_metrics.append(get_metrics_reg(y_back, preds=preds_y_back, name_set='backtest'))
        df_metrics.append(get_metrics_reg(y_out, preds=preds_y_out, name_set='out_of_sample'))

    df_metrics = pd.concat(df_metrics, axis=1).reset_index(names='metric')

    # Get importances DataFrame
    df_importances = get_feature_importance(mod, X_train, y_train, params_hypertuning)
    df_importances = df_importances.to_frame()
    

    # Create the model report
    df_specifications = pd.DataFrame({
        'date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'target': [target],
        'params_split': [str(
            {
                "station_fraq": subset_fraq,
                "n_months_backtest": n_months_backtest,
            }
        )],
        'algorithm': [algorithm],
        'params_algorithm': [str(params_algorithm)],
        'num_variables': [len(X_train.columns)]
    })

    dict_summary = {
        'specifications': df_specifications,
        'metrics': df_metrics,
        'importances': df_importances
    }

    return (
        df_final_preds, 
        dict_summary, 
        mod_isotonic
    )

#——————————————————————————————————————————————
# PREDICT FUNCTION
#——————————————————————————————————————————————

def get_x_y(df, ignore_cols, target=None):
    '''Returns the spliting into features (X), descriptive variables for target (y_full) and target(y)'''
    X = df[[col for col in df.columns if col not in ignore_cols]].copy()
    y_full = df[[col for col in df.columns if col in ignore_cols]].copy()
    if target is not None:
        y = df[target].copy()
    else:
        y = None
    return X, y_full, y




def assign_alert_level(prob: float | pd.Series, alert_thresholds) -> str | pd.Series:
    """Map probability value(s) to alert level: 'Red', 'Orange', or 'None'."""
    def _scalar(p: float) -> str:
        if p >= alert_thresholds["Red"]:
            return "Red"
        elif p >= alert_thresholds["Orange"]:
            return "Orange"
        return "None"

    if isinstance(prob, pd.Series):
        return prob.apply(_scalar)
    return _scalar(prob)


def predict_target(
    df_mt_orig: pd.DataFrame,
    mod,
    params_prepare: dict,
    params_predict: dict,
) -> pd.DataFrame:
    """
    Predict target for a particular dataset
    Args:
        - df_mt_orig : (pd.DataFrame) DataFrame with the feature matrix and descriptive variables
        - mod : (pickle) Trained model to use for prediction
        - params_predict : (dict) Dictionary containing parameters for prediction
    Returns:
        - df_predict : (pd.DataFrame) DataFrame with final predictions
    """
    peril = params_prepare['peril']
    ignore_cols = params_prepare[peril]['ignore_cols']

    peril = params_predict['peril']
    alert_thresholds = params_predict['alert_thresholds'][peril]

    # Transform categorical variables
    categorical_cols = list(
        set(
            df_mt_orig.select_dtypes(include=['object', 'category', 'bool']).columns
        ) - set(ignore_cols)
    )
    for col in categorical_cols:
        df_mt_orig[col] = df_mt_orig[col].astype('category')


    # Split data into feature matrices, target dimensions and target variable
    X, y_full, y = get_x_y(df_mt_orig, ignore_cols)

    cols_to_use = mod.feature_names_in_.tolist()
    X = X[cols_to_use].copy()

    # Get the predicted values
    preds_y = mod.predict_proba(X)[:, 1]
    y_full['pred_target'] = preds_y

    y_full['alert_level'] =  assign_alert_level(y_full["pred_target"], alert_thresholds)

    return y_full

#————————————————————————————————————————————————
# APPEND SPATIAL PARAMETERS
#————————————————————————————————————————————————


def build_alerts_with_admin(
    forecast_df: pd.DataFrame,
    geometry_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Filter forecast_df to rows with alert_level in `alert_levels`,
    spatial-join with geometry_gdf to append department/province/municipality,
    and return a clean DataFrame ready for CSV export.

    Parameters
    ----------
    forecast_df  : DataFrame with lon, lat, predict_prob, alert_level, time
    geometry_gdf : GeoDataFrame with department, province, municipality, geometry
    alert_levels : levels to keep; defaults to ["Orange", "Red"]

    Returns
    -------
    DataFrame with columns:
        lon, lat, time, predict_prob, alert_level,
        department, province, municipality
    """
    alert_levels = ['Orange', 'Red']

    filtered = forecast_df[forecast_df["alert_level"].isin(alert_levels)].copy()

    if filtered.empty:
        return pd.DataFrame(columns=[
            "idPolygon", "dept", "prov", "mun", "lon", "lat", 
            "time", "pred_target", "alert_level",
        ])


    # Convert points to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        filtered,
        geometry=gpd.points_from_xy(filtered["lon"], filtered["lat"]),
        crs="EPSG:4326",
    )

    admin_cols = ["idPolygon", "dept", "prov", "mun", "geometry"]
    joined = gpd.sjoin(
        points_gdf,
        geometry_gdf[admin_cols], 
        how="left",
        predicate="within",
    )

    out_cols = [
        "idPolygon", "dept", "prov", "mun", "lon", "lat", 
        "time", "pred_target", "alert_level",
    ]
    df_alerts = joined[out_cols]
    df_alerts = df_alerts[(df_alerts['mun']!='OTRO') & (~df_alerts['mun'].isna())].copy()
    return df_alerts.reset_index(drop=True)