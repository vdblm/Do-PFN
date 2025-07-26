from datasets import load_dataset
from scripts.transformer_prediction_interface.base import DoPFNRegressor
from scripts.tabular_metrics.regression import normalized_mean_squared_error_metric, root_mean_squared_error_metric
import numpy as np

dataset = load_dataset(ds_name='sales') # 'law_race'
dopfn = DoPFNRegressor()

train_ds, test_ds = dataset.generate_valid_split(split_number=1, n_splits=5)

dopfn.fit(train_ds.x, train_ds.y)
t_int = test_ds.x[:, 0]

cate_pred = dopfn.predict_cate(test_ds.x)
cate_true = test_ds.cate

print('MSE:', float(normalized_mean_squared_error_metric(cate_true, cate_pred)))
print('PEHE:', float(root_mean_squared_error_metric(cate_true, cate_pred)))


