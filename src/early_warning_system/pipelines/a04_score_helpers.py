from .utils import *


from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.base import clone
from sklearn.metrics import get_scorer, make_scorer
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, median_absolute_error, max_error
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, precision_recall_fscore_support, jaccard_score, recall_score
from sklearn.inspection import permutation_importance

from scipy.stats import rankdata, spearmanr, ks_2samp

#—————————————————————————————————————————
# CUSTOM SCORER FUNCTIONS
#—————————————————————————————————————————


def precision_at_target_recall(y_true, y_proba, target_recall=0.8):
    """Return the maximum precision achievable at or above target recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # precision_recall_curve returns arrays sorted by recall decreasing
    mask = recall >= target_recall
    if not np.any(mask):
        return 0.0  # target recall not achievable
    
    # Among recalls >= target, take the max precision
    return np.max(precision[mask])

# Wrap in sklearn scorer
precision_at_recall_scorer = make_scorer(
    precision_at_target_recall,
    needs_proba=True,        # tell sklearn to pass predict_proba
    greater_is_better=True,
    target_recall=0.8
)


def ks_statistic_multiclass(classes, y, y_pred):
    list_classes = []
    list_ks = []
    for i, c in enumerate(classes):
        list_classes.append(c)
        list_ks.append(ks_2samp(y_pred[y!=c, i], y_pred[y==c, i]).statistic)
    
    df_metric = pd.DataFrame({
        'class': list_classes,
        'ks_statistic': list_ks
    })
    ks = {'results': df_metric, 'average': df_metric['ks_statistic'].mean()}
    return ks


def roc_auc_score_multiclass(classes, y, y_pred):
    list_classes = []
    list_roc = []
    for i, c in enumerate(classes):
        y_true = [1 if y_value==c else 0 for y_value in y]
        list_classes.append(c)
        list_roc.append(roc_auc_score(y_true, y_pred[:,i]))
    
    df_metric = pd.DataFrame({
        'class': list_classes,
        'roc_auc_score': list_roc
    })
    roc_auc = {'results': df_metric, 'average': df_metric['roc_auc_score'].mean()}
    return roc_auc  


def gini_score(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1

gini_scorer = make_scorer(gini_score, needs_proba=True)


def gini_score_multiclass(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    return 2 * auc - 1

gini_multiclass_scorer = make_scorer(gini_score_multiclass, needs_proba=True)


#———————————————————————————————————————————————
# METRIC FUNCTIONS FOR MODEL EVALUATION
#———————————————————————————————————————————————


def get_metrics_reg(y, preds=None, mod = None, X=None, name_set='train'):
    # We obtain the GINI for train and val
    if preds is None:
        preds = mod.predict(X)
    
    r2 = r2_score(y, preds)
    rmse = root_mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    meae = median_absolute_error(y, preds)
    maxe = max_error(y, preds)

    # Summary of results
    df_results = pd.DataFrame({
        f'{name_set}': [r2, rmse, mae, meae, maxe],
    }, index = ['R2', 'RMSE', 'MAE', 'MEAE', 'MAXE'])
    return df_results


def get_metrics_classif(y, preds=None, mod = None, X=None, name_set='train'):
    # We obtain the GINI for train and val
    if preds is None:
        preds = mod.predict_proba(X)[:,1]
    
    auc = round(roc_auc_score(y, preds), 4) * 100 #r2_score(y, preds)
    gini = 2*auc-100
    ks = round(ks_2samp(preds[y == 0], preds[y == 1]).statistic, 4)* 100
    preds_bin = np.where(preds>0.5, 1, 0)
    f1 = round(f1_score(y, preds_bin), 4) * 100
    rec = round(recall_score(y, preds_bin), 4) * 100
    prec = round(precision_at_target_recall(y, preds, 0.25)*100)

    # Summary of results
    df_results = pd.DataFrame({
        f'{name_set}': [auc, gini, ks, f1, rec, prec],
    }, index = ['AUC', 'GINI', 'KS', 'F1', 'REC', 'PR'])
    return df_results


def get_metrics_multiclassif(y, y_prob=None, y_pred=None, mod = None, X=None, name_set='train'):
    # We obtain the GINI for train and val
    if y_pred is None or y_prob is None:
        y_prob = mod.predict_proba(X)
        y_pred = mod.predict(X)
    
    classes = mod.classes_

    n_class = len(classes)

    auc = round(roc_auc_score_multiclass(classes, y, y_prob)['average'], 4) * 100 #r2_score(y, preds)
    gini = 2*auc - 100
    ks = round(ks_statistic_multiclass(classes, y, y_prob)['average'], 4)* 100
    prfs = precision_recall_fscore_support(y, y_pred)
    prec = round(prfs[0].mean() if n_class > 2 else prfs[0][1], 4) * 100
    recall = round(prfs[1].mean() if n_class > 2 else prfs[1][1], 4) * 100
    f1 = round(prfs[2].mean() if n_class > 2 else prfs[2][1], 4) * 100
    jaccard = round(jaccard_score(y, y_pred, average='macro' if n_class > 2 else 'binary'), 4)*100

    # Summary of results
    df_results = pd.DataFrame({
        f'{name_set}': [auc, gini, ks, prec, recall, f1, jaccard],
    }, index = ['AUC', 'GINI', 'KS', 'PREC', 'RECALL', 'F1', 'JACCARD'])
    
    return df_results


def get_feature_importance(mod, X_val, y_val, params):
    algorithm = params['algorithm']
    if algorithm == "hgbr":
        # Use permutation importance to calculate importance
        importances_result = permutation_importance(mod, X_val, y_val, n_repeats=4, random_state=24, n_jobs=-1)
        # The features are in the same order as they appear in X
        feature_importance = pd.Series(importances_result.importances_mean, index=X_val.columns)
    elif algorithm == "lgbr":
        importances = list(mod.feature_importances_/sum(mod.feature_importances_))
        feature_importance = pd.Series(importances, index=mod.feature_names_in_).sort_values(ascending=False)
    elif algorithm == "xgbr":
        importances = list(mod.feature_importances_/sum(mod.feature_importances_))
        feature_importance = pd.Series(importances, index=mod.feature_names_in_).sort_values(ascending=False)
    if algorithm == "hgbc":
        # Use permutation importance to calculate importance
        importances_result = permutation_importance(mod, X_val, y_val, n_repeats=4, random_state=24, n_jobs=-1)
        # The features are in the same order as they appear in X
        feature_importance = pd.Series(importances_result.importances_mean, index=X_val.columns)
    elif algorithm == "lgbc":
        importances = list(mod.feature_importances_/sum(mod.feature_importances_))
        feature_importance = pd.Series(importances, index=mod.feature_names_in_).sort_values(ascending=False)
    elif algorithm == "xgbc":
        importances = list(mod.feature_importances_/sum(mod.feature_importances_))
        feature_importance = pd.Series(importances, index=mod.feature_names_in_).sort_values(ascending=False)
    return feature_importance