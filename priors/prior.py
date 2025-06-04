from abc import ABCMeta, abstractmethod
from typing import Set, Optional, ClassVar
from dataclasses import dataclass, fields
import torch
from torch.utils.data import DataLoader
from functools import partial


def assert_no_nones(lis: list):
    assert all(
        e is not None for e in lis
    ), f"Merging attribute, where we don't know how to merge with Nones, {lis}"
    return lis


def triplet_tensor_merge_batches(attr, attr_name, batch_sizes, batch_dim=1):
    assert_no_nones(attr)
    assert all([a.shape[-1] == attr[0].shape[-1] for a in attr]), (
        f"Attr name: {attr_name} All tensors must have the same shape except for the first dimension."
        f" {attr_name} has shapes {[a.shape for a in attr]}"
    )
    return torch.cat(attr, batch_dim)


def list_merge(lists, f_name, batch_sizes):
    
    return sum(
        [
            list([None] * bs if sublist is None else sublist)
            for bs, sublist in zip(batch_sizes, lists)
        ],
        [],
    )


NOP = lambda *a: None


@dataclass
class Batch:
    """
    A batch of data, with non-optional x, y, and target_y attributes.
    All other attributes are optional.

    If you want to add an attribute for testing only, you can just assign it after creation like:
    ```
        batch = Batch(x=x, y=y, target_y=target_y)
        batch.test_attribute = test_attribute
    ```
    """

    x: torch.Tensor  # shape: (data_points, batch_size, features)
    x_merge_func: ClassVar = triplet_tensor_merge_batches

    y: torch.Tensor  # shape: (data_points, batch_size)
    y_merge_func: ClassVar = triplet_tensor_merge_batches

    target_y: torch.Tensor  # shape: (data_points, batch_size)
    target_y_merge_func: ClassVar = triplet_tensor_merge_batches

    x_int: torch.Tensor  # shape: (data_points, batch_size, features)
    x_int_merge_func: ClassVar = triplet_tensor_merge_batches
    
    def other_filled_attributes(
        self, set_of_attributes: Set[str] = frozenset(("x", "y", "target_y"))
    ):
        return [
            f.name
            for f in fields(self)
            if f.name not in set_of_attributes and getattr(self, f.name) is not None
        ]


def merge_batches(*batches, ignore_attributes=[]):
    """
    Merge all supported non-None fields in a pre-specified (general) way in batch dimesnsion,
    e.g. mutliple batch.x are concatenated in the batch dimension.
    :param ignore_attributes: attributes to remove from the merged batch, treated as if they were None.
    :return:
    """
    fields_to_be_merged = [
        f.name
        for f in fields(batches[0])
        if f.name not in ignore_attributes
        and any(getattr(b, f.name) is not None for b in batches)
    ]

    batch_sizes = [b.x.shape[1] for b in batches]

    # TODO: Check that fields does not return the merge funcs
    merge_funcs = {
        f.name: Batch.__dict__[f"{f.name}_merge_func"]
        for f in fields(batches[0])
        if f.name not in ignore_attributes
    }

    assert all(
        f in merge_funcs for f in fields_to_be_merged
    ), f"Unknown fields encountered in `safe_merge_batches_in_batch_dim`, {fields_to_be_merged}."
    return Batch(
        **{
            f: merge_funcs[f]([getattr(batch, f) for batch in batches], f, batch_sizes)
            for f in fields_to_be_merged
        }
    )
