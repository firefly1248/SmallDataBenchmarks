import warnings, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from utils import load_data_df
import importlib.util, numpy as np

spec = importlib.util.spec_from_file_location('bench', 'optuna_models.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# numeric dataset
X, y, c = load_data_df('iris')
print(f'iris: shape={X.shape}, cat_cols={c}')

# categorical dataset
X2, y2, c2 = load_data_df('abalone-3class')
print(f'abalone: shape={X2.shape}, cat_cols={c2}')

# CatFeaturesEncoder — ordinal
enc = m.CatFeaturesEncoder('ordinal')
Xt = enc.fit_transform(X2, y2)
print(f'After ordinal encoding: object cols remaining = {Xt.select_dtypes("object").shape[1]}')

# CatFeaturesEncoder — target
enc2 = m.CatFeaturesEncoder('target')
Xt2 = enc2.fit_transform(X2, y2)
print(f'After target encoding: object cols remaining = {Xt2.select_dtypes("object").shape[1]}')

# CatBoostNativeWrapper
cb = m.CatBoostNativeWrapper(c2, n_estimators=5,
                              loss_function='MultiClass', verbose=0, random_state=0)
cb.fit(X2, y2)
preds = cb.predict_proba(X2)
print(f'CatBoostNativeWrapper predict_proba shape: {preds.shape}')

print('All OK')
