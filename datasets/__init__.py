from __future__ import annotations

import copy
import pandas as pd

import torch
import numpy as np
import warnings
from typing import Optional,  Literal

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pickle as pkl
from copy import deepcopy

class DatasetModifications:
    def __init__(self, classes_capped: bool, feats_capped: bool, samples_capped: bool):
        """
        :param classes_capped: Whether the number of classes was capped
        :param feats_capped: Whether the number of features was capped
        :param samples_capped: Whether the number of samples was capped
        """
        self.classes_capped = classes_capped
        self.feats_capped = feats_capped
        self.samples_capped = samples_capped


class TabularDataset:
    def __init__(
        self,
        name: str,
        x: torch.tensor,
        y: torch.tensor,
        task_type: str,
        attribute_names: list[str],
        categorical_feats: Optional[list[int]] = None,
        modifications: Optional[DatasetModifications] = None,
        splits: Optional[list[tuple[torch.tensor, torch.tensor]]] = None,
        benchmark_name: Optional[str] = None, #TODO -Jake
        extra_info: Optional[dict] = None,
        description: Optional[str] = None,
    ):
        """
        :param name: Name of the dataset
        :param x: The data matrix
        :param y: The labels
        :param categorical_feats: A list of indices of categorical features
        :param attribute_names: A list of attribute names
        :param modifications: A DatasetModifications object
        :param splits: A list of splits, each split is a tuple of (train_indices, test_indices)
        """
        if categorical_feats is None:
            categorical_feats = []

        self.name = name
        self.x = x
        self.y = y
        self.categorical_feats = categorical_feats
        self.attribute_names = attribute_names
        self.modifications = (
            modifications
            if modifications is not None
            else DatasetModifications(
                classes_capped=False, feats_capped=False, samples_capped=False
            )
        )
        self.splits = splits
        self.task_type = task_type
        self.benchmark_name = benchmark_name
        self.extra_info = extra_info
        self.description = description

        if self.task_type in ("multiclass", "fairness_multiclass"):
            from model.encoders import MulticlassClassificationTargetEncoder

            self.y = MulticlassClassificationTargetEncoder.flatten_targets(self.y)

    def get_dataset_identifier(self):
        if self.task_type in ("fairness_multiclass", "fairness_regression", "do_regression"):
            return self.name
        else:
            tid = (
                    self.extra_info["openml_tid"]
                    if "openml_tid" in self.extra_info
                    else "notask"
                )
            did = (
                self.extra_info["openml_did"]
                if "openml_did" in self.extra_info
                else self.extra_info["did"]
            )
            return f"{did}_{tid}"

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(self.x.numpy(), columns=self.attribute_names)
        df = df.astype({name: "category" for name in self.categorical_names})
        df["target"] = self.y.numpy()
        return df

    @property
    def categorical_names(self) -> list[str]:
        return [self.attribute_names[i] for i in self.categorical_feats]

    def infer_and_set_categoricals(self) -> None:
        """
        Infers and sets categorical features from the data and sets the categorical_feats attribute. Don't use this
        method if the categorical indicators are already known from a predefined source. This method is used to infer
        categorical features from the data itself and only an approximation.
        """
        dummy_df = pd.DataFrame(self.x.numpy(), columns=self.attribute_names)
        encoded_with_categoricals = infer_categoricals(
            dummy_df,
            max_unique_values=20,
            max_percentage_of_all_values=0.1,
        )

        categorical_idxs = [
            i
            for i, dtype in enumerate(encoded_with_categoricals.dtypes)
            if dtype == "category"
        ]
        self.categorical_feats = categorical_idxs

    def __getitem__(self, indices):
        # convert a simple index x[y] to a tuple for consistency
        # if not isinstance(indices, tuple):
        #    indices = tuple(indices)
        ds = copy.deepcopy(self)
        ds.x = ds.x[indices]
        ds.y = ds.y[indices]

        if self.task_type == "fairness_multiclass":
            ds.prot_attrs = ds.prot_attrs[indices]
            if ds.dowhy_data is not None:
                ds.dowhy_data['df'] = ds.dowhy_data['df'].iloc[indices]
            if 'counterfactual' in ds.name:
                ds.dowhy_data['df_cntf'] = ds.dowhy_data['df_cntf'].iloc[indices]

        elif self.task_type == "do_regression":
            ds.x_obs = ds.x_obs[indices]
            ds.x_int = ds.x_int[indices]
            ds.y_obs = ds.y_obs[indices]
            ds.y_int = ds.y_int[indices]
            ds.cate = ds.cate[indices]

        return ds

    @staticmethod
    def check_is_valid_split(task_type, ds, index_train, index_test):
        if task_type not in ("multiclass", "fairness_multiclass"):
            return True

        # Checks if the set of classes are the same in dataset and its subsets
        if set(torch.unique(ds.y[index_train]).tolist()) != set(
            torch.unique(ds.y).tolist()
        ):
            return False
        if set(torch.unique(ds.y[index_test]).tolist()) != set(
            torch.unique(ds.y).tolist()
        ):
            return False

        return True

    def generate_valid_split(
        self,
        n_splits: int | None = None,
        splits: list[list[list[int], list[int]]] | None = None,
        split_number: int = 1,
        auto_fix_stratified_splits: bool = False,
    ) -> tuple[TabularDataset, TabularDataset] | tuple[None, None]:
        """Generates a deterministic train-(test/valid) split.

        Both splits must contain the same classes and all classes in the entire datasets.
        If no such split can be sampled, returns None.

        :param splits: A list of splits, each split is a tuple of (train_indices, test_indices) or None. If None, we generate the splits.
        :param n_splits: The number of splits to generate. Only required if splits is None.
        :param split_number: The split id. n_splits are coming from the same split and are disjoint. Further splits are
            generated by changing the seed. Only used if splits is None.
        :auto_fix_stratified_splits: If True, we try to fix the splits if they are not valid. Only used if splits is None.

        :return: the train and test split in format of TabularDataset or None, None if no valid split could be generated.
        """     

        if split_number == 0:
            raise ValueError("Split number 0 is not used, we index starting from 1.")
        # We are using split numbers from 1 to 5 to legacy reasons
        split_number = split_number - 1

        if splits is None:
            if n_splits is None:
                raise ValueError("If `splits` is None, `n_splits` must be set.")
            # lazy import as not needed elsewhere.
            from utils import get_cv_split_for_data

            # assume torch tensor as nothing else possible according to typing.
            x = self.x if isinstance(self.x, np.ndarray) else self.x.numpy()
            y = self.y if isinstance(self.y, np.ndarray) else self.y.numpy()

            splits, *_ = get_cv_split_for_data(
                x=x,
                y=y,
                n_splits=n_splits,
                splits_seed=(split_number // n_splits)
                + 1,  # deterministic for all splits from one seed/split due to using //
                stratified_split=self.task_type in ("multiclass", "fairness_multiclass"),
                safety_shuffle=False,  # if ture, shuffle in the split function, and you have to update x and y
                auto_fix_stratified_splits=auto_fix_stratified_splits,
            )
            if isinstance(splits, str):
                print(f"Valid split could not be generated {self.name} due to {splits}")
                return None, None

            split_number_parsed = split_number % n_splits
            train_inds, test_inds = splits[split_number_parsed]
            train_ds = self[train_inds]
            test_ds = self[test_inds]   

            if self.task_type == "do_regression":
                train_ds.x = train_ds.x_obs
                train_ds.y = train_ds.y_obs
                test_ds.y = test_ds.y_int
                test_ds.x = deepcopy(test_ds.x_obs)
                test_ds.x[:, 0] = test_ds.x_int[:, 0] # simulate intervention
        else:
            train_inds, test_inds = splits[split_number]
            train_ds = self[train_inds]
            test_ds = self[test_inds]

            if self.task_type == "do_regression":
                train_ds.x = train_ds.x_obs
                train_ds.y = train_ds.y_obs
                test_ds.y = test_ds.y_int
                test_ds.x = deepcopy(test_ds.x_obs)
                test_ds.x[:, 0] = test_ds.x_int[:, 0] # simulate intervention

        return train_ds, test_ds

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def get_duplicated_samples(self, features_only=False, var_thresh=0.999999):
        """
        Calculates duplicated samples based on the covariance matrix of the data.

        :param features_only: Whether to only consider the features for the calculation
        :param var_thresh: The threshold for the variance to be considered a duplicate

        :return: Tuple of ((covariance matrix, duplicated_samples indices), fraction_duplicated_samples)
        """
        from utils import normalize_data

        if features_only:
            data = self.x.clone()
        else:
            data = torch.cat([self.x, self.y.unsqueeze(1)], dim=1)
        data[torch.isnan(data)] = 0.0

        x = normalize_data(data.transpose(1, 0))
        cov_mat = torch.cov(x.transpose(1, 0))
        cov_mat = torch.logical_or(cov_mat == 1.0, cov_mat > var_thresh).float()
        duplicated_samples = cov_mat.sum(axis=0)

        return (duplicated_samples, cov_mat), (
            duplicated_samples > 0
        ).float().mean().item()

class InterventionalDataset(TabularDataset):
    def __init__(self, 
                 x_obs: torch.Tensor,
                 y_obs: torch.Tensor,
                 x_int: torch.Tensor,
                 y_int: torch.Tensor,
                 do_scm: any,
                 attribute_names: List[str],
                 name: str = "", 
                 function_args: dict = None,
                 num_treatments: int = 1,
                 **kwargs):
        
        """
        Interventional dataset for causal inference tasks.
        :param x_obs: Observed features, shape (N_samples, N_features)
        :param y_obs: Observed labels, shape (N_samples,)
        :param x_int: Interventional features, shape (N_samples, N_features)
        :param y_int: Interventional labels, shape (N_samples,)
        :param scm: Structural causal model used to generate the data
        :param attribute_names: List of feature names
        :param name: Name of the dataset
        :param num_treatments: Number of treatments. ENSURE THAT THE TREATMENT VARIABLES ARE AT THE FRONT OF THE DATASET
        :param kwargs: Additional keyword arguments
        """

        # some assertions 
        assert x_obs.shape[0] == y_obs.shape[0], "x_obs and y_obs must have the same number of samples"
        assert x_int.shape[0] == y_int.shape[0], "x_int and y_int must have the same number of samples"
        assert x_obs.shape[1] == x_int.shape[1], "x_obs and x_int must have the same number of features"
        
        self.x_obs = x_obs.float()
        self.x_int = x_int.float()
        self.attribute_names = attribute_names
        self.y_obs = y_obs.float()
        self.y_int = y_int.float()
        self.do_scm = do_scm
        self.function_args = function_args
        
        super().__init__(
            task_type="do_regression", 
            x=torch.tensor(x_obs).float(), 
            y=torch.tensor(y_obs).float(), 
            attribute_names=attribute_names, 
            name=name)

    def get_splits(self):
        idcs = np.random.permutation(np.arange(len(self.y)))
        train_idcs, test_idcs = idcs[:200], idcs[200:]
        train_ds, test_ds = self.generate_valid_split(splits=[train_idcs, test_idcs])
        return train_ds, test_ds

def load_semi_real_interventional_datasets():
    semi_real_datasets = ['sales', 'law_race']

    datasets_to_return = []
    for i in tqdm.tqdm(range(len(semi_real_datasets)), desc=f'Semi-Real Benchmarks'):
        dataset = semi_real_datasets[i]
        data_path = Path('data/semi_real_new') / dataset

        with open(str(data_path) + f'/{dataset}.pkl', 'rb') as f:
            ds = pkl.load(f)
            datasets_to_return.append(ds)
        
    return datasets_to_return

import tqdm

def load_prior_sampling_casestudies(n_max: int = 10):
    synthetic_casestudies = ['Observed_Confounder', 
                             'Backdoor_Criterion',  
                             'Frontdoor_Criterion',
                             'Observed_Mediator',
                             'Observed_Mediator_and_Confounder',
                             'Unobserved_Confounder',
                             'Common_Effect',
                             'Observed_Confounder_Small']
                
    casestudy_benchmarks = []

    for casestudy in synthetic_casestudies:
        data_path = Path('/work/dlclarge1/robertsj-dopfn/Do-PFN/data/prior_new') / casestudy
        for i in tqdm.tqdm(range(1, n_max+1), desc=f'{casestudy} Benchmarks'):
            with open(str(data_path) + f'/{casestudy}_{i}.pkl', 'rb') as f:
                casestudy_benchmarks.append(pkl.load(f))

    return casestudy_benchmarks


def load_dataset(ds_name):
    task_type = "do_regression"

    if "datasets_dict" not in locals():
        datasets_dict = {}

    datasets_dict[f"valid_{task_type}"], df = get_benchmark_for_task(
        task_type,
        split="valid",
        max_samples=2000,
        return_as_lists=False,
        sel=False
    )

    dataset_map = {}
    for dataset in datasets_dict[f"valid_{task_type}"]:
        dataset_map[dataset.name] = dataset

    dataset = dataset_map[ds_name]

    return dataset

def get_benchmark_for_task(
    task_type: Literal["do_regression"],
    split: Literal["train", "valid", "debug", "test", "kaggle"] = "test",
    max_samples: Optional[int] = 10000,
    max_features: Optional[int] = 85,
    max_classes: Optional[int] = 2,
    max_num_cells: Optional[int] = None,
    min_samples: int = 50,
    filter_for_nan: bool = False,
    return_capped: bool = False,
    return_as_lists: bool = True,
    n_max: int = 200,
    load_data: bool = True,
    fairness_enabled: bool = False,
    sel=True,
) -> tuple[list[pd.DataFrame], pd.DataFrame | None]:
        
    if task_type == "do_regression":
        datasets = load_semi_real_interventional_datasets()
        datasets +=  load_prior_sampling_casestudies()
        return datasets, None

    else:
        raise NotImplementedError(f"Unknown task type {task_type}")

