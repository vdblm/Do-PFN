import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Dict
from torch.distributions.laplace import Laplace
import torch.distributions as dist


def activation_sampling():
    """
    Function to sample a random activation function from a predefined list.
    The sampled activation function is then applied to a given input tensor.

    :return: A randomly selected activation function from the list.
    """

    def identity(x):
        return x

    activations = [
        torch.square,
        torch.relu,
        torch.tanh,
        identity,
    ]
    
    return np.random.choice(activations)

def make_additive_noise_sampling(shape: tuple, noise_std: float = None) -> Callable[[], torch.Tensor]:
    """
    Generates a function that samples additive noise from a normal distribution.
    """
    def sample_noise():
        return torch.normal(0, noise_std, shape)
    
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

class MakeStructuralEquationsConstantNoise(nn.Module):
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
                 activation_sampling = activation_sampling,
                ) -> None:
        super().__init__()
        self.parents: List[str] = parents

        self.linear: nn.Linear = nn.Linear(len(parents), 1, bias=False) if len(parents) > 0 else None
        
        # self.linear.weight = torch.nn.Parameter(torch.normal(0, 1, self.linear.weight.shape))

        self.activation: Callable[[torch.Tensor], torch.Tensor] = activation_sampling()
        self.samples_shape: tuple = samples_shape

        self.additive_noise: torch.Tensor = make_additive_noise_sampling(shape=samples_shape, noise_std=noise_std)()
        # self.additive_noise: torch.Tensor = make_distribution_mixture_loc_scale_one_dist_per_feature(shape=samples_shape, std=noise_std)()

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
                output = self.linear(parent_tensor).squeeze(-1)
                output = self.activation(output)
                output += self.additive_noise

        return output
    
class MakeMakeStructuralEquationsConstantNoiseSquareNonlinearity(MakeStructuralEquationsConstantNoise):
    """
    A subclass of MakeStructuralEquationsConstantNoise that uses a square non-linearity.
    """

    def __init__(self, parents: List[str], samples_shape: tuple, noise_std: float) -> None:
        super().__init__(parents=parents, samples_shape=samples_shape, noise_std=noise_std)
        self.activation = torch.square  # Use square as the activation function



class MakeMakeStructuralEquationsConstantNoiseTanHNonlinearity(MakeStructuralEquationsConstantNoise):
    """
    A subclass of MakeStructuralEquationsConstantNoise that uses a square non-linearity.
    """

    def __init__(self, parents: List[str], samples_shape: tuple, noise_std: float) -> None:
        super().__init__(parents=parents, samples_shape=samples_shape, noise_std=noise_std)
        self.activation = torch.tanh


def make_distribution_mixture_loc_scale_one_dist_per_feature(
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