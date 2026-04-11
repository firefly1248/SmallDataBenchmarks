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
assert len(c) == 0

# categorical dataset
X2, y2, c2 = load_data_df('abalone-3class')
print(f'abalone: shape={X2.shape}, cat_cols={c2}')
assert 'V1' in c2

# ordinal encoding removes all object cols
enc = m.CatFeaturesEncoder('ordinal')
Xt = enc.fit_transform(X2, y2)
assert Xt.select_dtypes('object').shape[1] == 0, "ordinal encoding left object cols!"
print(f'ordinal: object cols after = {Xt.select_dtypes("object").shape[1]}  OK')

# target encoding
enc2 = m.CatFeaturesEncoder('target')
Xt2 = enc2.fit_transform(X2, y2)
assert Xt2.select_dtypes('object').shape[1] == 0, "target encoding left object cols!"
print(f'target:  object cols after = {Xt2.select_dtypes("object").shape[1]}  OK')

# dataset with no cats — encoder should be a no-op
enc3 = m.CatFeaturesEncoder('ordinal')
Xt3 = enc3.fit_transform(X, y)
print(f'iris (no cats): shape unchanged = {Xt3.shape == X.shape}  OK')

print('\nAll fast checks OK')
