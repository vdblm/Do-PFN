import torch


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