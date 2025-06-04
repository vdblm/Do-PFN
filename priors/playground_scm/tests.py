import CausalPlayground
from priors.playground_scm.generators import CausalGraphGenerator, SCMGenerator
from priors.playground_scm.scm import StructuralCausalModel
import networkx as nx
import random
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time

from priors.playground_scm.generators import SCMGenerator
from priors.playground_scm.MakeStructuralEquationsConstantNoise import MakeStructuralEquationsConstantNoise
from priors.playground_scm.MakeStructuralEquationsConstantNoise import make_additive_noise_sampling, make_exo_dist_samples

def torch_random_choice(a: torch.Tensor, size: int = 1, replace: bool = False) -> torch.Tensor:
    """
    A wrapper around torch.randint to mimic numpy's random.choice behavior.
    :param a: The array to choose from.
    :param size: The number of samples to draw.
    :param replace: Whether to sample with replacement.
    :return: A tensor of random choices.
    """
    if replace:
        return a[torch.randint(0, len(a), (size,))]
    else:
        return a[torch.randperm(len(a))[:size]]


def run_test() -> float:
    """
    Run a test to check if the observational and interventional datasets are consistent.
    Returns the time to run the data sampling process.
    :return: time_delta: time taken to run the test
    """
    
    n_endo, n_exo = 4, 2
    num_features = 2
    batch_size, num_samples = 4, 100
    samples_shape = (batch_size, num_samples)
    seed = torch.randint(0, 10000, (1,)).item()
    random.seed(seed)
    
    t0 = time.time()
    gen = SCMGenerator(all_functions={'nonlinear': MakeStructuralEquationsConstantNoise}, seed=seed, samples_shape=samples_shape)

    scm = gen.create_random(possible_functions=["nonlinear"], n_endo=n_endo, n_exo=n_exo,
                                            exo_distribution=make_exo_dist_samples(samples_shape),
                                            exo_distribution_kwargs={},
                                            allow_exo_confounders=True,
                                            noise_distribution = make_exo_dist_samples(samples_shape),
                                            noise_distribution_kwargs = {})[0]

    #scm.draw_graph() # visualize the graph

    # sample treatment and outcome from the graph
    t_choice = []
    graph = scm.create_graph()

    for var in list(graph.nodes)[:n_endo+n_exo]: 
        if graph.out_degree(var) > 0:  # check if the variable has any descendants
            t_choice.append(var)

    scm.t_key = torch_random_choice(t_choice)

    # sample outcome as descendent of treatment
    t_desc = list(nx.descendants(graph, scm.t_key))
    scm.y_key = torch_random_choice(t_desc)

    # sample observational dataset and binarize the treatment
    endo_obs, exo_obs = scm.get_next_sample(binarize=True, graph=graph)
    sample_obs = endo_obs | exo_obs

    # sample interventional dataset and binarize the treatment
    coin_flips = torch.randint(0, 2, (batch_size, num_samples))
    t1s_exp = scm.t1s.unsqueeze(1).expand(-1, num_samples)
    t2s_exp = scm.t2s.unsqueeze(1).expand(-1, num_samples)
    t_int = torch.where(coin_flips == 0, t1s_exp, t2s_exp)

    if 'X' in scm.t_key:  # if enogenous variable, change functional mechanism
        scm.do_interventions([(scm.t_key, (lambda: t_int, {}))])
    else:  # if exogenous variable, change the value of the variable directly
        exo_obs[scm.t_key] = t_int

    endo_int, exo_int = scm.get_next_sample(exogenous_vars=exo_obs, graph=graph)
    sample_int = endo_int | exo_int


    X_cand = set(graph.nodes) - set([scm.y_key, scm.t_key])
    X_keys = np.random.choice(list(X_cand), size=num_features, replace=False)

    x_obs = torch.stack([sample_obs[key] for key in X_keys]).permute(-1, 1, 0)
    x_int = torch.stack([sample_int[key] for key in X_keys]).permute(-1, 1, 0)

    y_obs, y_int = sample_obs[scm.y_key].T, sample_int[scm.y_key].T

    t1 = time.time()
    time_delta = t1 - t0

    dfs_obs = []
    for b in range(batch_size):
        df = pd.DataFrame()
        for key in list(sample_obs.keys()):
            df[key] = sample_obs[key][b].detach()
        dfs_obs.append(df)

    dfs_int = []
    for b in range(batch_size):
        df = pd.DataFrame()
        for key in list(sample_int.keys()):
            df[key] = sample_int[key][b].detach()
        dfs_int.append(df)

    for d_obs, d_int in zip(dfs_obs, dfs_int):

        # non nan values anywhere 
        assert d_obs.notna().all().all(), "There are NaN values in the observational dataset"
        assert d_int.notna().all().all(), "There are NaN values in the interventional dataset"

        # if the treatment is the same then everything else should be the same
        assert (d_obs == d_int)[d_obs[scm.t_key] == d_int[scm.t_key]].all().all()
        
        # non-treatment exogenous variables should be the same
        exo_keys = ['U'+str(i) for i in range(n_exo)]
        non_treatment_exogenous = list(set(exo_keys) - set([scm.t_key]))
        assert d_obs[non_treatment_exogenous].equals(d_int[non_treatment_exogenous]), "Non-treatment exogenous variables are not the same in observational and interventional dataset"

        # non-descendents of the treatment should be the same
        endo_keys = ['X'+str(i) for i in range(n_exo)]
        non_treatment_descendents = list(set(endo_keys) - set(t_desc) - set([scm.t_key]))
        
        # Check for numerical errors by allowing a small tolerance
        if len(non_treatment_descendents) > 0:
            tolerance = 1e-10
            diff = d_obs[non_treatment_descendents].subtract(d_int[non_treatment_descendents]).abs()
            max_diff = diff.max().max()
            if max_diff >= tolerance:
                error_location = diff.stack().idxmax()  # Find the row and column where the maximum difference occurs
                raise AssertionError(f"Non-treatment descendents are not the same within numerical tolerance in observational and interventional dataset. "
                                    f"Max difference {max_diff} at row {error_location[0]}, column {error_location[1]}")

        # treatment variables should be in some cases different if the outcome is a descendent of the treatment
        # Right now, this is always the case 
        assert not d_obs[scm.t_key].equals(d_int[scm.t_key]), "Treatment variables are the same in observational and interventional dataset"

    
    return time_delta


def run_n_tests(n:int) -> tuple[float, float]:
    """
    run n tests to check if the observational and interventional datasets are consistent.
    Returns the time to run the data sampling process.
    :param n: number of tests to run
    :return:
    mean: mean time taken to run the tests
    se: standard error of the mean time taken to run the tests
    """
    t_vals = []

    for i in tqdm(range(n)):
        t_i =  run_test()
        t_vals.append(t_i)
    
    mean = np.mean(t_vals)
    se = np.std(t_vals) / np.sqrt(n)

    return mean, se


if __name__ == "__main__":
    n = 100
    time_delta, time_delta_se = run_n_tests(n)
    print(f"Average time to sample one batch: {time_delta:.5f} ± {time_delta_se:.5f} seconds. Total time for {n} sampled batches: {time_delta*n:.5f} seconds")
