from datasets import load_dataset
from scripts.transformer_prediction_interface.base import DoPFNRegressor
import numpy as np

dataset = load_dataset(ds_name='sales')
dopfn = DoPFNRegressor()

train_ds, test_ds = dataset.generate_valid_split(n_splits=2)

dopfn.fit(train_ds.x, train_ds.y)
y_pred = dopfn.predict_full(test_ds.x)

