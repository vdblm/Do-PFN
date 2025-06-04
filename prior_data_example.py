import pickle as pkl
from priors.doscm import get_batch

hyperparameters = {'num_unobserved': 1,
                   'seed': 42,
                   'noise_std': 0.01,
                   'exo_std': 0.1,
                   'graph': None,
                   't_idx': None, 
                   'y_idx': None,
                   'x_idcs': None,
                   'zero_one_treatment': True,
                   'inference_cov': 'pre_interventional',
                   'test': False
    }

prior_ds, do_scm = get_batch(
    batch_size=1,
    hyperparameters=hyperparameters,
    return_SCM=True,
    seq_len=100,
    num_features=3
)