from utils import default_device

import torch
from priors import Batch
import networkx as nx
import random
import numpy as np
from priors.playground_scm.generators import SCMGenerator
from priors.playground_scm.MakeStructuralEquationsConstantNoise import MakeStructuralEquationsConstantNoise, MakeMakeStructuralEquationsConstantNoiseSquareNonlinearity, MakeMakeStructuralEquationsConstantNoiseTanHNonlinearity
from priors.playground_scm.MakeStructuralEquationsConstantNoise import make_additive_noise_sampling, make_exo_dist_samples, make_distribution_mixture_loc_scale_one_dist_per_feature
from priors.playground_scm.utils_playground_scm import torch_random_choice
import pickle as pkl

def get_batch(
    batch_size: int,
    seq_len: int, 
    num_features: int,
    hyperparameters: int,
    device: str = default_device,
    num_outputs: int = 1,
    num_treatments: int = 1,    
    epoch=None,
    return_SCM = False, # if True, return the SCM object as well
    set_nonlinearity = None,
    **kwargs,
):

    class DoSCM(torch.nn.Module):
        def __init__(self, hyperparameters):
            super(DoSCM, self).__init__()

            with torch.no_grad():
                for key in hyperparameters:
                    setattr(self, key, hyperparameters[key])

            self.batch_size = batch_size
            self.num_samples = seq_len
            self.samples_shape = (self.batch_size, self.num_samples)
            self.num_nodes = num_features + self.num_unobserved + num_outputs + num_treatments

            edge_prob_min = 1 / (num_features + 1)
            self.edge_prob = np.random.uniform(edge_prob_min, 1)
        
        def forward(self):  

            gen = SCMGenerator(all_functions={'nonlinear': MakeStructuralEquationsConstantNoise}, 
                               seed=self.seed, samples_shape=self.samples_shape, noise_std=self.noise_std)
            
            
            if set_nonlinearity is not None: 
                if set_nonlinearity == "square":
                    gen = SCMGenerator(all_functions={'nonlinear': MakeMakeStructuralEquationsConstantNoiseSquareNonlinearity}, 
                               seed=self.seed, samples_shape=self.samples_shape, noise_std=self.noise_std)
                if set_nonlinearity == "tanh":
                    gen = SCMGenerator(all_functions={'nonlinear': MakeMakeStructuralEquationsConstantNoiseTanHNonlinearity}, 
                               seed=self.seed, samples_shape=self.samples_shape, noise_std=self.noise_std)

            # generate graph
            if self.graph is None:
                self.graph = gen.create_graph_from_nodes(num_nodes=self.num_nodes, p=self.edge_prob)

            # call gen.create_scm_from_graph
            self.scm = gen.create_scm_from_graph(self.graph, 
                                            possible_functions=["nonlinear"], 
                                            # exo_distribution=make_exo_dist_samples(self.samples_shape, self.exo_std), 
                                            exo_distribution=make_distribution_mixture_loc_scale_one_dist_per_feature(self.samples_shape, self.exo_std),
                                            exo_distribution_kwargs={})
            
            self.scm.zero_one_treatment = self.zero_one_treatment
            scm_graph = self.scm.create_graph()

            """
            scm = gen.create_random(possible_functions=["nonlinear"], n_endo=self.n_endo, n_exo=self.n_exo,
                                                    exo_distribution=make_exo_dist_samples(self.samples_shape),
                                                    exo_distribution_kwargs={},
                                                    allow_exo_confounders=True)[0]
            """

            # sample treatment and outcome from the graph
            if len(list(scm_graph.edges)) == 0:
                self.scm.t_key = torch_random_choice(list(scm_graph.nodes))
            elif self.t_idx is None:
                t_choice = []
                for var in list(scm_graph.nodes): 
                    if scm_graph.out_degree(var) > 0:  # check if the variable has any descendants
                        t_choice.append(var)
                self.scm.t_key = torch_random_choice(t_choice)
            else:
                for var in list(scm_graph.nodes):
                    if str(self.t_idx) in var:
                        self.scm.t_key = var

            # sample outcome as descendent of treatment
            if len(list(scm_graph.edges)) == 0:
                self.scm.y_key = torch_random_choice(list(set(scm_graph.nodes)-set([self.scm.t_key])))
            elif self.y_idx is None:
                t_desc = list(nx.descendants(scm_graph, self.scm.t_key))
                self.scm.y_key = torch_random_choice(t_desc)
            else:
                for var in list(scm_graph.nodes):
                    if str(self.y_idx) in var:
                        self.scm.y_key = var

            # sample observational dataset and binarize the treatment
            endo_obs, exo_obs = self.scm.get_next_sample(binarize=True, graph=scm_graph)
            sample_obs = endo_obs | exo_obs

            # prepare intervention or counterfactual 
            if self.inference_cov == 'counterfactual':
                t_cntf = sample_obs[self.scm.t_key]
                for b in range(self.batch_size):
                    t_obs = sample_obs[self.scm.t_key][b]
                    t_cntf[b] = torch.where(t_obs == self.scm.t1s[b], self.scm.t2s[b], self.scm.t1s[b])

                if 'X' in self.scm.t_key:  # if enogenous variable, change functional mechanism
                    self.scm.do_interventions([(self.scm.t_key, (lambda: t_cntf, {}))])
                else:  # if exogenous variable, change the value of the variable directly
                    exo_obs[self.scm.t_key] = t_cntf

            elif self.inference_cov == 'pre_interventional' or self.inference_cov == 'pre_treatment':
                # sample interventional dataset and binarize the treatment
                coin_flips = torch.randint(0, 2, (batch_size, self.num_samples))
                t1s_exp = self.scm.t1s.unsqueeze(1).expand(-1, self.num_samples)
                t2s_exp = self.scm.t2s.unsqueeze(1).expand(-1, self.num_samples)
                t_int = torch.where(coin_flips == 0, t1s_exp, t2s_exp)
                
                if 'X' in self.scm.t_key:  # if enogenous variable, change functional mechanism
                    self.scm.do_interventions([(self.scm.t_key, (lambda: t_int, {}))])
                else:  # if exogenous variable, change the value of the variable directly
                    exo_obs[self.scm.t_key] = t_int

            endo_int, exo_int = self.scm.get_next_sample(exogenous_vars=exo_obs, graph=scm_graph)
            sample_int = endo_int | exo_int

            self.scm.undo_interventions()

            if self.x_idcs is None:
                X_cand = set(scm_graph.nodes) - set([self.scm.y_key, self.scm.t_key])
                X_keys = [self.scm.t_key] + list(np.random.choice(list(X_cand), size=num_features, replace=False))
            else:
                X_keys = [self.scm.t_key]
                for var in list(scm_graph.nodes):
                    if int(var[-1]) in self.x_idcs:
                        X_keys.append(var)

            self.x_keys = X_keys  # save the keys for later use

            x_obs = torch.stack([sample_obs[key] for key in X_keys]).permute(-1, 1, 0)
            x_int = torch.stack([sample_int[key] for key in X_keys]).permute(-1, 1, 0)

            y_obs, y_int = sample_obs[self.scm.y_key].T, sample_int[self.scm.y_key].T 

            if self.test:
                with open('/work/dlclarge1/robertsj-dopfn/prior-fitting/data/prior_sampling/Test/Test_1.pkl', 'rb') as f:
                    ds = pkl.load(f)
                if self.batch_size == 1:
                    return ds.x_obs[:, :1 ,:], ds.x_int[:, :1 ,:], ds.y_obs[:, 0, :], ds.y_int[:, 0, :]
                else:
                    return ds.x_obs, ds.x_int, ds.y_obs, ds.y_int
            else:
                return x_obs, x_int, y_obs, y_int

    do_scm = DoSCM(hyperparameters).to(device)
    x_obs, x_int, y_obs, y_int = do_scm.forward()

    y_obs = y_obs.detach().unsqueeze(-1)
    y_int = y_int.detach().unsqueeze(-1)
    x_obs = x_obs.detach()
    x_int = x_int.detach()

    if torch.any(torch.isnan(x_int)) or torch.any(torch.isnan(x_obs)) or torch.any(torch.isnan(y_obs)) or torch.any(torch.isnan(y_int)): 
        y_obs[:] = -100
        y_int[:] = -100
        x_obs[:] = -100
        x_int[:] = -100

    if return_SCM:
        return Batch(x=x_obs, y=y_obs, target_y=y_int, x_int=x_int), do_scm
    else:
        return Batch(x=x_obs, y=y_obs, target_y=y_int, x_int=x_int)
