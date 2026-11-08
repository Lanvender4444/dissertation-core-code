"""Numeric histogram-boosting calibration exported to portable JSON trees.

Only training needs scikit-learn==1.7.2. Inference uses NumPy/SciPy; no detector
training or pickle loading. This standard learner is NOT a claimed invention.
"""
import numpy as np
from scipy.special import expit


def export(estimator):
    if list(estimator.classes_)!=[0,1]:
        raise ValueError('Only binary output supported')
    trees = []
    for iteration in estimator._predictors:
        if len(iteration)!=1:
            raise ValueError('Only one binary tree per iteration supported')
        nodes = iteration[0].nodes
        if nodes['is_categorical'].any():
            raise ValueError('Categorical export unsupported')
        tree = {k:nodes[k].tolist() for k in ('value','feature_idx','num_threshold','left','right','is_leaf')}
        if not np.isfinite(tree['value']).all() or not np.isfinite(tree['num_threshold']).all():
            raise ValueError('Nonfinite exported tree')
        trees.append(tree)
    return dict(dim=int(estimator.n_features_in_),baseline=float(estimator._baseline_prediction[0,0]),trees=trees,
        representation='Numeric binary trees, leaf values already include learning rate; finite features only')


def compile_model(model):
    return dict(dim=model['dim'],baseline=model['baseline'],trees=[{k:np.asarray(v,int if k in
        ('feature_idx','left','right','is_leaf') else float) for k,v in tree.items()} for tree in model['trees']])


def predict(xs,model):
    x = np.asarray(xs,float)
    if not len(x):
        return np.empty(0)
    if x.ndim!=2 or x.shape[1]!=model['dim'] or not np.isfinite(x).all():
        raise ValueError('Invalid portable predictor input')
    eta = np.full(len(x),model['baseline'],float)
    for tree in model['trees']:
        position = np.zeros(len(x),int)
        while True:
            active = np.flatnonzero(np.asarray(tree['is_leaf'])[position]==0)
            if not len(active):
                break
            indices = position[active]
            feature = np.asarray(tree['feature_idx'])[indices]
            left = x[active,feature]<=np.asarray(tree['num_threshold'])[indices]
            position[active] = np.where(left,np.asarray(tree['left'])[indices],np.asarray(tree['right'])[indices])
        eta += np.asarray(tree['value'])[position]
    return expit(eta)


def fit(xs,ys,params):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    if sklearn.__version__!='1.7.2':
        raise ValueError('Frozen training library version required')
    x,y = np.asarray(xs,float),np.asarray(ys,int)
    if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or sorted(set(y))!=[0,1]:
        raise ValueError('Invalid binary calibration data')
    with threadpool_limits(limits=1):
        estimator = HistGradientBoostingClassifier(**params).fit(x,y)
        result = export(estimator)
        actual = estimator.predict_proba(x)[:,1]
    error = float(np.max(np.abs(actual-predict(x,compile_model(result)))))
    if error>1e-12:
        raise ValueError('Portable export differs from training library')
    return dict(result,params=params,training_library='scikit-learn==1.7.2',samples=len(y),positive=int(y.sum()),
        training_export_max_error=error,target='Post-hoc corrected GT-hit/novelty proxy, not calibrated physical existence probability')
