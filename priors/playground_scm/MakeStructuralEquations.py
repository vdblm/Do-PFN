import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Dict
from torch.distributions.laplace import Laplace
import torch.distributions as dist
import torch.nn.functional as F

class ToModule(nn.Module):
    def __init__(self, func):
        super(ToModule, self).__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

def activation_sampling(nonlins: str):
    """
    Sample once and return a fixed activation function f such that
    every call f(x) is deterministic—and for the ‘sophisticated’ variants
    we clamp outputs to [-1000,1000].
    """

    # stateless activations
    def identity(x): return x
    def neg(x):      return -x

    # summed builder
    def make_summed():
        pool = [torch.square, torch.relu, torch.tanh, identity]
        f1, f2 = np.random.choice(pool, size=2, replace=False)
        def f(x):
            r = (f1(x) + f2(x)) / 2
            r = torch.clamp(r, -1000, 1000)
            return r
        return f

    # sophisticated_sampling_1 builder
    def make_soph1():
        pool = [
            torch.nn.ReLU(), torch.nn.ReLU6(), torch.nn.SELU(), torch.nn.SiLU(),
            torch.nn.Softplus(), torch.nn.Hardtanh(), torch.sign, torch.sin,
            lambda x: torch.exp(-x**2), torch.exp,
            lambda x: torch.sqrt(torch.abs(x)),
            lambda x: (torch.abs(x) < 1).float(),
            lambda x: x**2, lambda x: torch.abs(x),
        ]
        r0 = np.random.rand()
        if r0 < 1/3:
            f0 = np.random.choice(pool)
            def f(x):
                r = f0(x)
                return torch.clamp(r, -1000, 1000)
        elif r0 < 2/3:
            f1_, f2_ = np.random.choice(pool, size=2, replace=False)
            w = np.random.rand(2); w /= w.sum()
            def f(x):
                r = w[0]*f1_(x) + w[1]*f2_(x)
                return torch.clamp(r, -1000, 1000)
        else:
            f1_, f2_, f3_ = np.random.choice(pool, size=3, replace=False)
            w = np.random.rand(3); w /= w.sum()
            def f(x):
                r = w[0]*f1_(x) + w[1]*f2_(x) + w[2]*f3_(x)
                return torch.clamp(r, -1000, 1000)
        return f

    # sophisticated_sampling_2 builder (with LayerNorm)
    def make_soph2():
        inner = make_soph1()
        def f(x):
            x = F.layer_norm(x, x.shape)
            r = inner(x)
            # inner already clamps, but just to be sure:
            return torch.clamp(r, -1000, 1000)
        return f

    # sophisticated_sampling_2_rescaling builder
    def make_soph2_rescale():
        inner = make_soph1()
        a = torch.randn(1)
        b = torch.randn(1)
        scale = torch.exp(2 * a)
        def f(x):
            x = F.layer_norm(x, x.shape)
            x = scale * (x + b)
            r = inner(x)
            return torch.clamp(r, -1000, 1000)
        return f


    if nonlins in ("mixed", "post"):
        pool = [torch.square, torch.relu, torch.tanh, identity]
        return np.random.choice(pool)

    elif nonlins == "tanh":
        return torch.tanh

    elif nonlins == "sin":
        return torch.sin

    elif nonlins == "neg":
        return neg
    
    elif nonlins == "id":
        return identity

    elif nonlins == "elu":
        return torch.nn.functional.elu

    elif nonlins == "summed":
        return make_summed()

    elif nonlins == "sophisticated_sampling_1":
        return make_soph1()

    elif nonlins == "sophisticated_sampling_1_normalization":
        return make_soph2()

    elif nonlins == "sophisticated_sampling_1_rescaling_normalization":
        return make_soph2_rescale()

    else:
        pool = [torch.square, torch.relu, torch.tanh, identity]
        return np.random.choice(pool)


def make_additive_noise_gaussian(shape: tuple, std: float = None) -> Callable[[], torch.Tensor]:
    """
    Generates a function that samples additive noise from a normal distribution.
    """
    def sample_noise():
        return torch.normal(0, std, shape)
    
    return sample_noise

def make_additive_noise_laplace(shape: tuple, std: float = None) -> Callable[[], torch.Tensor]:
    """
    Generates a function that samples additive noise from a laplace distribution.
    """
    def sample_noise():
        return 
    
    return sample_noise

def make_exo_dist_samples(shape: tuple, exo_std: float = None) -> Callable[[], torch.Tensor]:
    """
    Function to create samples from a uniform distribution for exogenous variables.
    The function uses the specified sample shape to generate the samples.

    :param sample_shape: Shape of the samples to be generated in one forward-pass in the SCM.
    :return: A function that generates samples from a uniform distribution.
    """
    def sample_exogenous():
        return torch.normal(0, exo_std, shape)
    
    return sample_exogenous

class MakeStructuralEquations(nn.Module):
    """
    A PyTorch module that defines a structural equation for a node in a causal graph 
    based on its parents. The model linearly combines the parent values using a linear 
    layer and applies a randomly selected non-linear activation function.
    The additive noise is only sampled once and added to the output.


    :param parents: List of names of the parent variables for this node.
    :param possible_activations: Optional list of activation functions to sample from.
                                  If not provided, defaults to [square, ReLU, tanh].

    """ 

    def __init__(self, 
                 parents: List[str], 
                 samples_shape: tuple,
                 noise_std: float,
                 noise_dist: str,
                 nonlins: str,
                 max_hidden_layers: int,
                ) -> None:
        super().__init__()
        self.parents: List[str] = parents
        
        if len(parents) > 0:
            self.layers: nn.Linear = nn.Linear(len(parents), 1, bias=False) 

        else:
            self.layers = None
            
        self.activation: Callable[[torch.Tensor], torch.Tensor] = activation_sampling(nonlins=nonlins)
        self.samples_shape: tuple = samples_shape
        self.nonlins: str = nonlins
        
        if noise_dist == 'gaussian':
            self.additive_noise: torch.Tensor = make_additive_noise_gaussian(shape=samples_shape, std=noise_std)()
        elif noise_dist == 'laplace': 
            self.additive_noise: torch.Tensor = make_additive_noise_mixed(shape=samples_shape, 
                                                                          std=noise_std,
                                                                          mixture_proportions=[0, 1, 0, 0])()
        elif noise_dist == 'student': 
            self.additive_noise: torch.Tensor = make_additive_noise_mixed(shape=samples_shape, 
                                                                          std=noise_std,
                                                                          mixture_proportions=[0, 0, 1, 0])()
        elif noise_dist == 'gumbel': 
            self.additive_noise: torch.Tensor = make_additive_noise_mixed(shape=samples_shape, 
                                                                          std=noise_std,
                                                                          mixture_proportions=[0, 0, 0, 1])()  

    def forward(self, **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the structural equation.

        :param kwargs: Keyword arguments where each key is a parent name and each value 
                       is a scalar or tensor representing the parent's value.

        :raises KeyError: If any required parent variable is missing from kwargs.

        :return: Transformed tensor after applying the learned linear combination 
                 and the sampled non-linear activation.
        """
        if len(self.parents) == 0:
           output = self.additive_noise
        else:
            parent_values = [kwargs[parent] for parent in self.parents]
            parent_tensor = torch.stack(parent_values, dim=-1)
            
            with torch.no_grad():
                if self.nonlins != 'post':
                    output = self.layers(parent_tensor).squeeze(-1)
                    output = self.activation(output)
                    output += self.additive_noise
                else:
                    output = self.layers(parent_tensor).squeeze(-1)
                    output += self.additive_noise
                    output = self.activation(output)
                    
        return output


def make_additive_noise_mixed(
    shape: tuple,
    std: float = None,
    distributions: list = [dist.Normal, dist.Laplace, dist.StudentT, dist.Gumbel],
    mixture_proportions: list = [1/4, 1/4, 1/4, 1/4],
    std2scale: dict = None
) -> Callable[[], torch.Tensor]:
    """
    Function to create samples from a mixture of distributions, one per feature (i.e. per batch element).
    All samples are of shape (batch_size, n_samples), one distribution randomly assigned per element in the batch.

    :param shape: Shape of the output tensor (batch_size, n_samples). Can also be (n_samples,).
    :param std: Standard deviation of the exogenous variable.
    :param distributions: List of distribution classes to be used in the mixture.
    :param mixture_proportions: List of proportions for each distribution in the mixture.
    :param std2scale: Dictionary mapping distribution classes to functions converting std to scale.
    :return: A function that generates samples from the specified mixture distribution.
    """
    student_t_df = 3.0  # ensure finite variance

    if len(shape) == 2:
        batch_size, n_samples = shape
    elif len(shape) == 1:  # if only one dimension is provided, assume it's the number of samples
        batch_size = 1
        n_samples = shape[0]

    if std2scale is None:
        std2scale = {
            dist.Normal: lambda std: std,
            dist.Laplace: lambda std: std / (2 ** 0.5),
            dist.StudentT: lambda std: std * ((student_t_df - 2) / student_t_df) ** 0.5,
            dist.Gumbel: lambda std: (6 ** 0.5 / torch.pi) * std,
        }

    assert len(distributions) == len(mixture_proportions), "Distributions and mixture proportions must match in length."
    assert len(std2scale) == len(distributions), "std2scale must have same length as distributions."
    assert len(shape) in [1, 2], "Shape must be either 1D or 2D. But got {}".format(len(shape))

    # Normalize proportions
    mixture_proportions = torch.tensor(mixture_proportions, dtype=torch.float32)
    mixture_proportions /= mixture_proportions.sum()

    # Initialize all distributions
    initialized_distributions = []
    for dist_class in distributions:
        if dist_class == dist.StudentT:
            scale = std2scale[dist_class](std)
            dist_inst = dist_class(df=student_t_df, loc=0.0, scale=scale)
        elif dist_class == dist.Gumbel:
            euler_gamma = 0.5772156649015329
            scale = std2scale[dist_class](std)
            loc = -euler_gamma * scale
            dist_inst = dist_class(loc=loc, scale=scale)
        else:
            scale = std2scale[dist_class](std)
            dist_inst = dist_class(loc=0.0, scale=scale)
        initialized_distributions.append(dist_inst)

    def sample():
        # Assign each batch element a distribution ID
        dist_indices = dist.Categorical(mixture_proportions).sample((batch_size,))
        dist_indices = dist_indices.unsqueeze(1).expand(-1, n_samples)

        res_sample = torch.zeros((batch_size, n_samples), dtype=torch.float32)
        for i, dist_inst in enumerate(initialized_distributions):
            sample = dist_inst.sample((batch_size, n_samples))
            res_sample[dist_indices == i] = sample[dist_indices == i]

        if len(shape) == 1:
            res_sample = res_sample.squeeze(0)
        return res_sample

    return sample