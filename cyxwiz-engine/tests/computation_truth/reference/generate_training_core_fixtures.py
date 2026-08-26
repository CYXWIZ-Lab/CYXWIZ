#!/usr/bin/env python3
"""Generate deterministic PyTorch fixtures for core CyxWiz training math."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "training_core_pytorch.json"
)


def tensor_fixture(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "values": value.reshape(-1).tolist(),
    }


def linear_case() -> dict[str, Any]:
    input_tensor = torch.tensor(
        [[1.0, 2.0, -1.0], [0.0, -3.0, 4.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.tensor(
        [[0.5, -1.0, 2.0], [-0.25, 0.75, 1.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    bias = torch.tensor(
        [0.1, -0.2], dtype=torch.float32, requires_grad=True
    )
    grad_output = torch.tensor(
        [[1.0, -0.5], [0.25, 2.0]], dtype=torch.float32
    )

    output = functional.linear(input_tensor, weight, bias)
    grad_input, grad_weight, grad_bias = torch.autograd.grad(
        output,
        (input_tensor, weight, bias),
        grad_outputs=grad_output,
    )

    return {
        "operation": "torch.nn.functional.linear",
        "dtype": "float32",
        "parameter_gradient_reduction": "sum_over_batch",
        "tolerance": {"atol": 1.0e-5, "rtol": 1.0e-5},
        "input": tensor_fixture(input_tensor),
        "weight": tensor_fixture(weight),
        "bias": tensor_fixture(bias),
        "grad_output": tensor_fixture(grad_output),
        "expected": {
            "output": tensor_fixture(output),
            "grad_input": tensor_fixture(grad_input),
            "grad_weight": tensor_fixture(grad_weight),
            "grad_bias": tensor_fixture(grad_bias),
        },
    }


def flatten_case(
    name: str,
    shape: list[int],
    start_dim: int,
) -> dict[str, Any]:
    element_count = math.prod(shape)
    input_tensor = (
        torch.arange(element_count, dtype=torch.float32)
        .reshape(shape)
        .sub(7.5)
        .requires_grad_(True)
    )
    module = torch.nn.Flatten(start_dim=start_dim)
    output = module(input_tensor)
    grad_output = torch.linspace(
        -1.25, 1.75, steps=element_count, dtype=torch.float32
    ).reshape(output.shape)
    (grad_input,) = torch.autograd.grad(
        output, input_tensor, grad_outputs=grad_output
    )

    return {
        "name": name,
        "operation": "torch.nn.Flatten",
        "dtype": "float32",
        "parameters": {"start_dim": start_dim, "end_dim": -1},
        "tolerance": {"atol": 0.0, "rtol": 0.0},
        "input": tensor_fixture(input_tensor),
        "grad_output": tensor_fixture(grad_output),
        "expected": {
            "output": tensor_fixture(output),
            "grad_input": tensor_fixture(grad_input),
        },
    }


def flatten_matrix() -> list[dict[str, Any]]:
    return [
        flatten_case("rank4_start1", [2, 1, 2, 2], 1),
        flatten_case("rank4_start2", [2, 1, 2, 2], 2),
        flatten_case("rank3_negative_start", [2, 2, 2], -2),
        flatten_case("rank2_start1_identity_shape", [2, 2], 1),
    ]


def tensor_shape_semantics() -> dict[str, Any]:
    input_tensor = torch.arange(6, dtype=torch.float32).reshape(1, 2, 1, 3)
    scalar = torch.tensor(7.0, dtype=torch.float32)
    empty = torch.empty((2, 0, 3), dtype=torch.float32)

    return {
        "operation": "torch.Tensor shape operations",
        "dtype": "float32",
        "input": tensor_fixture(input_tensor),
        "scalar": tensor_fixture(scalar),
        "empty": tensor_fixture(empty),
        "expected": {
            "reshape": tensor_fixture(input_tensor.reshape(3, 2)),
            "view": tensor_fixture(input_tensor.view(2, 3)),
            "squeeze_all": tensor_fixture(torch.squeeze(input_tensor)),
            "squeeze_last_non_singleton": tensor_fixture(
                torch.squeeze(input_tensor, -1)
            ),
            "squeeze_middle_non_singleton": tensor_fixture(
                torch.squeeze(input_tensor, 1)
            ),
            "squeeze_negative_singleton": tensor_fixture(
                torch.squeeze(input_tensor, -2)
            ),
            "unsqueeze_front": tensor_fixture(torch.unsqueeze(input_tensor, 0)),
            "unsqueeze_back": tensor_fixture(torch.unsqueeze(input_tensor, -1)),
            "flatten_all": tensor_fixture(torch.flatten(input_tensor)),
            "flatten_middle": tensor_fixture(torch.flatten(input_tensor, 1, 2)),
            "scalar_squeeze": tensor_fixture(torch.squeeze(scalar)),
            "scalar_squeeze_dim": tensor_fixture(torch.squeeze(scalar, -1)),
            "scalar_unsqueeze": tensor_fixture(torch.unsqueeze(scalar, -1)),
            "scalar_flatten": tensor_fixture(torch.flatten(scalar)),
            "empty_reshape": tensor_fixture(empty.reshape(0, 6)),
            "empty_flatten": tensor_fixture(torch.flatten(empty, 1, 2)),
        },
    }


def tensor_permute_case(
    name: str,
    shape: list[int],
    dims: list[int],
) -> dict[str, Any]:
    input_tensor = torch.arange(
        math.prod(shape), dtype=torch.float32
    ).reshape(shape)
    output = input_tensor.permute(tuple(dims))
    return {
        "name": name,
        "operation": "torch.Tensor.permute",
        "dims": dims,
        "input": tensor_fixture(input_tensor),
        "expected": tensor_fixture(output),
    }


def tensor_permute_matrix() -> list[dict[str, Any]]:
    return [
        tensor_permute_case("scalar", [], []),
        tensor_permute_case("rank1_identity", [6], [-1]),
        tensor_permute_case("rank2_swap", [2, 3], [1, 0]),
        tensor_permute_case("rank2_negative_swap", [2, 3], [-1, -2]),
        tensor_permute_case("rank3_rotate", [2, 3, 4], [2, 0, 1]),
        tensor_permute_case("rank3_negative", [2, 3, 4], [-2, -1, -3]),
        tensor_permute_case("rank4_general", [2, 2, 3, 2], [3, 1, 0, 2]),
        tensor_permute_case("rank4_identity", [2, 2, 3, 2], [0, 1, 2, 3]),
        tensor_permute_case("empty", [2, 0, 3], [2, 0, 1]),
    ]


def tensor_slice_case(
    name: str,
    shape: list[int],
    dim: int,
    start: int,
    end: int = -1,
    step: int = 1,
) -> dict[str, Any]:
    input_tensor = torch.arange(
        math.prod(shape), dtype=torch.float32
    ).reshape(shape)
    normalized_dim = dim if dim >= 0 else dim + len(shape)
    selection = [slice(None)] * len(shape)
    selection[normalized_dim] = slice(start, None if end == -1 else end, step)
    output = input_tensor[tuple(selection)]
    return {
        "name": name,
        "operation": "torch.Tensor.__getitem__",
        "dim": dim,
        "start": start,
        "end": end,
        "step": step,
        "input": tensor_fixture(input_tensor),
        "expected": tensor_fixture(output),
    }


def tensor_index_select_case(
    name: str,
    shape: list[int],
    dim: int,
    indices: list[int],
) -> dict[str, Any]:
    input_tensor = torch.arange(
        math.prod(shape), dtype=torch.float32
    ).reshape(shape)
    normalized_dim = dim if dim >= 0 else dim + len(shape)
    dim_size = shape[normalized_dim]
    oracle_indices = [
        index + dim_size if index < 0 else index
        for index in indices
    ]
    output = torch.index_select(
        input_tensor,
        normalized_dim,
        torch.tensor(oracle_indices, dtype=torch.int64),
    )
    return {
        "name": name,
        "operation": "torch.index_select",
        "dim": dim,
        "indices": indices,
        "oracle_indices": oracle_indices,
        "input": tensor_fixture(input_tensor),
        "expected": tensor_fixture(output),
    }


def tensor_indexing_matrix() -> dict[str, Any]:
    return {
        "slice": [
            tensor_slice_case("rank1_step", [6], 0, 1, -1, 2),
            tensor_slice_case("rank2_negative_bounds", [3, 4], -1, -3),
            tensor_slice_case("rank2_empty", [3, 4], 1, 3, 1),
            tensor_slice_case("rank3_step", [2, 3, 4], 1, 0, -1, 2),
            tensor_slice_case("rank4_middle", [2, 2, 3, 2], 2, 1, 3),
            tensor_slice_case("clamped_bounds", [6], 0, -99, 99),
            tensor_slice_case("explicit_negative_end", [6], 0, 1, -2),
            tensor_slice_case("zero_dimension", [2, 0, 3], 1, 0),
        ],
        "index_select": [
            tensor_index_select_case("rank1_repeat", [6], 0, [5, 1, 1, 0]),
            tensor_index_select_case("rank2_negative_dim", [3, 4], -1, [3, 1]),
            tensor_index_select_case("rank3_general", [2, 3, 4], 1, [2, 0]),
            tensor_index_select_case("rank4_general", [2, 2, 3, 2], 2, [2, 0, 2]),
            tensor_index_select_case("empty_indices", [3, 4], 1, []),
            tensor_index_select_case("negative_extension", [3, 4], 1, [-1, 0]),
            tensor_index_select_case("zero_other_dimension", [2, 0, 3], 0, [1, 0]),
        ],
    }


def tensor_values(shape: list[int], offset: int = 0) -> torch.Tensor:
    return torch.arange(
        offset, offset + math.prod(shape), dtype=torch.float32
    ).reshape(shape)


def tensor_cat_case(
    name: str,
    shapes: list[list[int]],
    dim: int,
) -> dict[str, Any]:
    tensors: list[torch.Tensor] = []
    offset = 0
    for shape in shapes:
        tensor = tensor_values(shape, offset)
        tensors.append(tensor)
        offset += tensor.numel()
    return {
        "name": name,
        "operation": "torch.cat",
        "dim": dim,
        "inputs": [tensor_fixture(tensor) for tensor in tensors],
        "expected": tensor_fixture(torch.cat(tensors, dim=dim)),
    }


def tensor_stack_case(
    name: str,
    shape: list[int],
    count: int,
    dim: int,
) -> dict[str, Any]:
    element_count = math.prod(shape)
    tensors = [
        tensor_values(shape, index * element_count)
        for index in range(count)
    ]
    return {
        "name": name,
        "operation": "torch.stack",
        "dim": dim,
        "inputs": [tensor_fixture(tensor) for tensor in tensors],
        "expected": tensor_fixture(torch.stack(tensors, dim=dim)),
    }


def tensor_split_size_case(
    name: str,
    shape: list[int],
    split_size: int,
    dim: int,
) -> dict[str, Any]:
    input_tensor = tensor_values(shape)
    outputs = torch.split(input_tensor, split_size, dim=dim)
    return {
        "name": name,
        "operation": "torch.split",
        "dim": dim,
        "split_size": split_size,
        "input": tensor_fixture(input_tensor),
        "expected": [tensor_fixture(output) for output in outputs],
    }


def tensor_split_sizes_case(
    name: str,
    shape: list[int],
    sizes: list[int],
    dim: int,
) -> dict[str, Any]:
    input_tensor = tensor_values(shape)
    outputs = torch.split(input_tensor, sizes, dim=dim)
    return {
        "name": name,
        "operation": "torch.split",
        "dim": dim,
        "sizes": sizes,
        "input": tensor_fixture(input_tensor),
        "expected": [tensor_fixture(output) for output in outputs],
    }


def tensor_chunk_case(
    name: str,
    shape: list[int],
    chunks: int,
    dim: int,
) -> dict[str, Any]:
    input_tensor = tensor_values(shape)
    outputs = torch.chunk(input_tensor, chunks, dim=dim)
    return {
        "name": name,
        "operation": "torch.chunk",
        "dim": dim,
        "chunks": chunks,
        "input": tensor_fixture(input_tensor),
        "expected": [tensor_fixture(output) for output in outputs],
    }


def tensor_concat_matrix() -> dict[str, Any]:
    return {
        "cat": [
            tensor_cat_case("rank1", [[2], [3]], 0),
            tensor_cat_case("rank2_negative_dim", [[2, 1], [2, 2]], -1),
            tensor_cat_case("rank3_middle", [[2, 1, 2], [2, 2, 2]], 1),
            tensor_cat_case(
                "rank4_middle", [[1, 2, 1, 2], [1, 2, 2, 2]], 2
            ),
            tensor_cat_case("empty_concat_axis", [[2, 0, 3], [2, 2, 3]], 1),
            tensor_cat_case("rank1_empty_identity", [[0], [2, 3]], 0),
            tensor_cat_case("zero_other_axis", [[2, 0, 1], [2, 0, 2]], 2),
        ],
        "stack": [
            tensor_stack_case("scalar", [], 2, 0),
            tensor_stack_case("rank1_last", [3], 2, -1),
            tensor_stack_case("rank2_middle", [2, 3], 2, 1),
            tensor_stack_case("rank3_to_rank4", [2, 2, 2], 2, -1),
            tensor_stack_case("empty", [2, 0], 2, 0),
        ],
        "split_size": [
            tensor_split_size_case("uneven", [2, 5], 2, 1),
            tensor_split_size_case("oversized", [3], 8, 0),
            tensor_split_size_case("empty_dimension", [0, 3], 2, 0),
        ],
        "split_sizes": [
            tensor_split_sizes_case("zero_section", [4], [1, 0, 3], 0),
            tensor_split_sizes_case("negative_dim", [2, 4], [1, 3], -1),
        ],
        "chunk": [
            tensor_chunk_case("uneven", [7], 3, 0),
            tensor_chunk_case("more_chunks_than_elements", [3], 5, 0),
            tensor_chunk_case("empty_dimension", [0, 3], 3, 0),
            tensor_chunk_case("negative_dim", [2, 5], 3, -1),
        ],
    }


def tensor_reduction_input(
    shape: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    count = math.prod(shape)
    if dtype == torch.uint8:
        return torch.tensor(
            [(index % 4) + 1 for index in range(count)], dtype=dtype
        ).reshape(shape)
    values = torch.arange(count, dtype=torch.float64).reshape(shape)
    values = (values.remainder(7) - 3).add(values * 0.125)
    return values.to(dtype)


def tensor_reduction_result(
    input_tensor: torch.Tensor,
    operation: str,
    dim: int | None,
    keepdim: bool,
    correction: int,
) -> torch.Tensor:
    dimension_args: dict[str, Any] = {}
    if dim is not None:
        dimension_args = {"dim": dim, "keepdim": keepdim}

    if operation == "sum":
        return torch.sum(
            input_tensor, dtype=input_tensor.dtype, **dimension_args
        )
    if operation == "prod":
        return torch.prod(
            input_tensor, dtype=input_tensor.dtype, **dimension_args
        )
    if operation == "max":
        return torch.amax(input_tensor, **dimension_args)
    if operation == "min":
        return torch.amin(input_tensor, **dimension_args)

    real_dtype = (
        torch.float64
        if input_tensor.dtype == torch.float64
        else torch.float32
    )
    real_input = input_tensor.to(real_dtype)
    if operation == "mean":
        return torch.mean(real_input, **dimension_args)
    if operation == "var":
        return torch.var(
            real_input, correction=correction, **dimension_args
        )
    if operation == "std":
        return torch.std(
            real_input, correction=correction, **dimension_args
        )
    raise ValueError(f"unsupported Tensor reduction: {operation}")


def tensor_reduction_case(
    name: str,
    shape: list[int],
    dtype: torch.dtype,
    operation: str,
    dim: int | None = None,
    keepdim: bool = False,
    correction: int = 0,
) -> dict[str, Any]:
    input_tensor = tensor_reduction_input(shape, dtype)
    return tensor_reduction_case_from_input(
        name, input_tensor, operation, dim, keepdim, correction
    )


def tensor_reduction_case_from_input(
    name: str,
    input_tensor: torch.Tensor,
    operation: str,
    dim: int | None = None,
    keepdim: bool = False,
    correction: int = 0,
) -> dict[str, Any]:
    expected = tensor_reduction_result(
        input_tensor, operation, dim, keepdim, correction
    )
    return {
        "name": name,
        "operation": f"torch.{operation}",
        "input_dtype": str(input_tensor.dtype).removeprefix("torch."),
        "output_dtype": str(expected.dtype).removeprefix("torch."),
        "dim": dim,
        "keepdim": keepdim,
        "correction": correction,
        "input": tensor_fixture(input_tensor),
        "expected": tensor_fixture(expected),
        "tolerance": {
            "atol": 1.0e-10 if expected.dtype == torch.float64 else 1.0e-5,
            "rtol": 1.0e-10 if expected.dtype == torch.float64 else 1.0e-5,
        },
    }


def tensor_reduction_matrix() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    dtype_matrix = [
        ("f32", torch.float32, [2, 3, 4]),
        ("f64", torch.float64, [2, 2, 2, 3]),
        ("i32", torch.int32, [2, 3]),
        ("i64", torch.int64, [2, 3, 2]),
        ("u8", torch.uint8, [2, 2, 3]),
    ]
    for dtype_name, dtype, shape in dtype_matrix:
        cases.extend([
            tensor_reduction_case(
                f"{dtype_name}_sum_global", shape, dtype, "sum"
            ),
            tensor_reduction_case(
                f"{dtype_name}_sum_negative_dim_keepdim",
                shape, dtype, "sum", -1, True
            ),
            tensor_reduction_case(
                f"{dtype_name}_mean_global", shape, dtype, "mean"
            ),
            tensor_reduction_case(
                f"{dtype_name}_mean_negative_dim_keepdim",
                shape, dtype, "mean", -1, True
            ),
            tensor_reduction_case(
                f"{dtype_name}_max_global", shape, dtype, "max"
            ),
            tensor_reduction_case(
                f"{dtype_name}_max_dim0", shape, dtype, "max", 0
            ),
            tensor_reduction_case(
                f"{dtype_name}_min_global", shape, dtype, "min"
            ),
            tensor_reduction_case(
                f"{dtype_name}_min_negative_dim",
                shape, dtype, "min", -1
            ),
            tensor_reduction_case(
                f"{dtype_name}_prod_global", shape, dtype, "prod"
            ),
            tensor_reduction_case(
                f"{dtype_name}_prod_dim0_keepdim",
                shape, dtype, "prod", 0, True
            ),
            tensor_reduction_case(
                f"{dtype_name}_var_global_population",
                shape, dtype, "var", correction=0
            ),
            tensor_reduction_case(
                f"{dtype_name}_var_negative_dim_sample",
                shape, dtype, "var", -1, False, 1
            ),
            tensor_reduction_case(
                f"{dtype_name}_std_global_population",
                shape, dtype, "std", correction=0
            ),
            tensor_reduction_case(
                f"{dtype_name}_std_negative_dim_sample",
                shape, dtype, "std", -1, False, 1
            ),
        ])

    cases.extend([
        tensor_reduction_case(
            "f32_sum_middle_dim", [2, 3, 4], torch.float32, "sum", 1
        ),
        tensor_reduction_case(
            "f32_var_middle_dim_keepdim_correction2",
            [2, 3, 4], torch.float32, "var", 1, True, 2
        ),
        tensor_reduction_case(
            "f64_std_global_negative_correction",
            [2, 2, 2, 3], torch.float64, "std", correction=-1
        ),
        tensor_reduction_case(
            "scalar_mean_dim0", [], torch.float32, "mean", 0, True
        ),
        tensor_reduction_case_from_input(
            "i32_sum_global_overflow",
            torch.tensor([torch.iinfo(torch.int32).max, 1], dtype=torch.int32),
            "sum",
        ),
        tensor_reduction_case_from_input(
            "i32_sum_axis_overflow",
            torch.tensor(
                [
                    [torch.iinfo(torch.int32).max, 1],
                    [torch.iinfo(torch.int32).min, -1],
                ],
                dtype=torch.int32,
            ),
            "sum", 1,
        ),
        tensor_reduction_case_from_input(
            "i32_prod_global_overflow",
            torch.tensor([torch.iinfo(torch.int32).max, 2], dtype=torch.int32),
            "prod",
        ),
        tensor_reduction_case_from_input(
            "i32_prod_axis_overflow",
            torch.tensor(
                [
                    [torch.iinfo(torch.int32).max, 2],
                    [torch.iinfo(torch.int32).min, -1],
                ],
                dtype=torch.int32,
            ),
            "prod", 1,
        ),
    ])
    return cases


def tensor_elementwise_input(
    shape: list[int],
    dtype: torch.dtype,
    offset: int,
) -> torch.Tensor:
    count = math.prod(shape)
    values = [((index + offset) % 5) + 1 for index in range(count)]
    return torch.tensor(values, dtype=dtype).reshape(shape)


def tensor_elementwise_case(
    name: str,
    operation: str,
    left: torch.Tensor,
    right: torch.Tensor | float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, Any]:
    if operation == "add":
        expected = left + right
    elif operation == "subtract":
        expected = left - right
    elif operation == "multiply":
        expected = left * right
    elif operation == "divide":
        expected = left / right
    elif operation == "pow":
        expected = torch.pow(left, right)
    elif operation == "sqrt":
        expected = torch.sqrt(left)
    elif operation == "exp":
        expected = torch.exp(left)
    elif operation == "log":
        expected = torch.log(left)
    elif operation == "abs":
        expected = torch.abs(left)
    elif operation == "sign":
        expected = torch.sign(left)
    elif operation == "clip":
        expected = torch.clamp(left, minimum, maximum)
    elif operation == "negate":
        expected = torch.neg(left)
    else:
        raise ValueError(f"unsupported Tensor elementwise operation: {operation}")

    result = {
        "name": name,
        "operation": operation,
        "input_dtype": str(left.dtype).removeprefix("torch."),
        "output_dtype": str(expected.dtype).removeprefix("torch."),
        "input": tensor_fixture(left),
        "expected": tensor_fixture(expected),
        "tolerance": {
            "atol": 1.0e-10 if expected.dtype == torch.float64 else 1.0e-5,
            "rtol": 1.0e-10 if expected.dtype == torch.float64 else 1.0e-5,
        },
    }
    if isinstance(right, torch.Tensor):
        result["rhs_kind"] = "tensor"
        result["rhs_dtype"] = str(right.dtype).removeprefix("torch.")
        result["rhs"] = tensor_fixture(right)
    elif right is not None:
        result["rhs_kind"] = "scalar"
        result["scalar"] = float(right)
    else:
        result["rhs_kind"] = "none"
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    return result


def tensor_elementwise_matrix() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    dtypes = [
        ("f32", torch.float32),
        ("f64", torch.float64),
        ("i32", torch.int32),
        ("i64", torch.int64),
        ("u8", torch.uint8),
    ]

    # Every public dtype pair exercises PyTorch promotion and rank-3
    # right-aligned broadcasting for the four tensor arithmetic operators.
    for left_name, left_dtype in dtypes:
        for right_name, right_dtype in dtypes:
            left = tensor_elementwise_input([2, 1, 3], left_dtype, 0)
            right = tensor_elementwise_input([1, 2, 1], right_dtype, 1)
            for operation in ["add", "subtract", "multiply", "divide"]:
                cases.append(tensor_elementwise_case(
                    f"tensor_{operation}_{left_name}_{right_name}_broadcast",
                    operation,
                    left,
                    right,
                ))

    # Pow has its own promotion/output contract and uses the same complete
    # dtype-pair broadcast matrix with small exact exponents.
    for left_name, left_dtype in dtypes:
        for right_name, right_dtype in dtypes:
            left = tensor_elementwise_input([2, 1, 3], left_dtype, 0)
            right = torch.tensor([1, 2], dtype=right_dtype).reshape(1, 2, 1)
            cases.append(tensor_elementwise_case(
                f"tensor_pow_{left_name}_{right_name}_broadcast",
                "pow",
                left,
                right,
            ))

    for dtype_name, dtype in dtypes:
        scalar_input = tensor_elementwise_input([2, 3], dtype, 0)
        for operation in ["add", "subtract", "multiply", "divide"]:
            cases.append(tensor_elementwise_case(
                f"scalar_{operation}_{dtype_name}",
                operation,
                scalar_input,
                2.5,
            ))
        cases.append(tensor_elementwise_case(
            f"scalar_pow_{dtype_name}", "pow", scalar_input, 2.0
        ))

        positive = tensor_elementwise_input([2, 3], dtype, 0)
        signed_values = (
            torch.tensor([-3, 0, 2, -1, 4, 1], dtype=dtype).reshape(2, 3)
            if dtype != torch.uint8
            else torch.tensor([0, 1, 2, 3, 4, 255], dtype=dtype).reshape(2, 3)
        )
        for operation in ["sqrt", "exp", "log"]:
            cases.append(tensor_elementwise_case(
                f"unary_{operation}_{dtype_name}", operation, positive
            ))
        for operation in ["abs", "sign", "negate"]:
            cases.append(tensor_elementwise_case(
                f"unary_{operation}_{dtype_name}", operation, signed_values
            ))
        cases.append(tensor_elementwise_case(
            f"unary_clip_{dtype_name}",
            "clip",
            signed_values,
            minimum=-1.5,
            maximum=2.5,
        ))

    # Rank and empty-shape sentinels complement the rank-3 dtype matrix.
    rank_shapes = [[], [3], [2, 3], [1, 2, 1, 3], [2, 0, 3]]
    for index, shape in enumerate(rank_shapes):
        left = tensor_elementwise_input(shape, torch.float32, 0)
        scalar = torch.tensor(2.0, dtype=torch.float32)
        cases.append(tensor_elementwise_case(
            f"rank_{index}_add_scalar_tensor", "add", left, scalar
        ))

    return cases


def dropout_semantics() -> dict[str, Any]:
    input_tensor = torch.tensor(
        [[1.0, -2.0, 3.5], [4.0, -5.0, 6.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    grad_output = torch.tensor(
        [[0.25, -0.5, 0.75], [1.0, -1.25, 1.5]],
        dtype=torch.float32,
    )

    eval_module = torch.nn.Dropout(0.5).eval()
    eval_output = eval_module(input_tensor)
    (eval_grad_input,) = torch.autograd.grad(
        eval_output, input_tensor, grad_outputs=grad_output
    )

    boundary_cases: list[dict[str, Any]] = []
    for probability in (0.0, 1.0):
        boundary_input = input_tensor.detach().clone().requires_grad_(True)
        boundary_output = torch.nn.Dropout(probability)(boundary_input)
        (boundary_grad_input,) = torch.autograd.grad(
            boundary_output, boundary_input, grad_outputs=grad_output
        )
        boundary_cases.append({
            "probability": probability,
            "expected": {
                "output": tensor_fixture(boundary_output),
                "grad_input": tensor_fixture(boundary_grad_input),
            },
        })

    distribution_cases: list[dict[str, Any]] = []
    sample_count = 65536
    for index, probability in enumerate((0.25, 0.5, 0.9)):
        torch.manual_seed(3901 + index)
        samples = torch.ones(
            sample_count, dtype=torch.float32, requires_grad=True
        )
        output = torch.nn.Dropout(probability)(samples)
        (grad_input,) = torch.autograd.grad(
            output, samples, grad_outputs=torch.ones_like(output)
        )
        distribution_cases.append({
            "probability": probability,
            "sample_count": sample_count,
            "expected_keep_scale": 1.0 / (1.0 - probability),
            "theoretical": {
                "zero_fraction": probability,
                "mean": 1.0,
                "variance": probability / (1.0 - probability),
            },
            "pytorch_observed": {
                "zero_fraction": float((output == 0).float().mean().item()),
                "mean": float(output.mean().item()),
                "variance": float(output.var(unbiased=False).item()),
                "backward_mask_mismatch_count": int(
                    (grad_input != output).sum().item()
                ),
            },
            "tolerance": {
                "zero_fraction": 0.015,
                "mean": 0.03,
                "variance": 0.3 if probability == 0.9 else 0.06,
            },
        })

    return {
        "operation": "torch.nn.Dropout",
        "dtype": "float32",
        "input": tensor_fixture(input_tensor),
        "grad_output": tensor_fixture(grad_output),
        "eval_expected": {
            "output": tensor_fixture(eval_output),
            "grad_input": tensor_fixture(eval_grad_input),
        },
        "boundary_cases": boundary_cases,
        "distribution_cases": distribution_cases,
        "rng_contract": (
            "Distribution parity is required across frameworks; exact mask "
            "replay is required only after resetting the same backend RNG seed."
        ),
    }


def gradient_accumulation_case(
    name: str,
    input_values: list[list[float]],
    target_values: list[int],
    microbatch_size: int,
    grad_accum_steps: int,
    class_weights: list[float] | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_reduction: str = "mean",
) -> dict[str, Any]:
    if loss_reduction not in {"mean", "sum"}:
        raise ValueError("gradient accumulation requires mean or sum reduction")
    inputs = torch.tensor(input_values, dtype=torch.float32)
    targets = torch.tensor(target_values, dtype=torch.int64)
    initial_weight = torch.tensor(
        [[0.2, -0.4], [-0.1, 0.3]], dtype=torch.float32
    )
    # CyxWiz Linear bias initialization is deterministically zero. These
    # fixtures use zero-valued features so the bias update is independent of
    # backend-specific random weight initialization while still exercising the
    # full Linear -> CrossEntropy -> accumulation -> SGD path.
    initial_bias = torch.zeros(2, dtype=torch.float32)
    model = torch.nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        model.weight.copy_(initial_weight)
        model.bias.copy_(initial_bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    weight = (
        torch.tensor(class_weights, dtype=torch.float32)
        if class_weights is not None
        else None
    )

    microbatches = list(
        zip(inputs.split(microbatch_size), targets.split(microbatch_size))
    )
    expected_steps: list[dict[str, Any]] = []
    correct_predictions = 0
    valid_predictions = 0
    epoch_loss_numerator = 0.0
    epoch_loss_denominator = 0.0
    for window_start in range(0, len(microbatches), grad_accum_steps):
        window = microbatches[window_start:window_start + grad_accum_steps]
        effective_inputs = torch.cat([batch[0] for batch in window])
        effective_targets = torch.cat([batch[1] for batch in window])
        optimizer.zero_grad(set_to_none=True)
        logits = model(effective_inputs)
        valid_mask = effective_targets != ignore_index
        correct_predictions += int(
            ((logits.argmax(dim=1) == effective_targets) & valid_mask)
            .sum()
            .item()
        )
        valid_predictions += int(valid_mask.sum().item())
        loss = functional.cross_entropy(
            logits,
            effective_targets,
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=loss_reduction,
        )
        loss.backward()
        optimizer.step()
        if loss_reduction == "sum":
            epoch_loss_numerator += float(loss.item())
            epoch_loss_denominator = 1.0
        else:
            semantic_denominator = (
                float(weight[effective_targets[valid_mask]].sum().item())
                if weight is not None
                else float(valid_mask.sum().item())
            )
            epoch_loss_numerator += float(loss.item()) * semantic_denominator
            epoch_loss_denominator += semantic_denominator
        expected_steps.append({
            "ending_microbatch": window_start + len(window),
            "window_microbatch_count": len(window),
            "window_sample_count": int(effective_inputs.shape[0]),
            "loss": float(loss.item()),
            "weight": tensor_fixture(model.weight),
            "bias": tensor_fixture(model.bias),
        })

    return {
        "name": name,
        "operation": "torch.nn.Linear + torch.nn.functional.cross_entropy + torch.optim.SGD",
        "dtype": "float32",
        "loss_reduction": loss_reduction,
        "class_weights": class_weights or [],
        "ignore_index": ignore_index,
        "label_smoothing": label_smoothing,
        "learning_rate": 0.1,
        "microbatch_size": microbatch_size,
        "grad_accum_steps": grad_accum_steps,
        "expected_optimizer_step_count": len(expected_steps),
        "expected_train_accuracy": (
            correct_predictions / valid_predictions
            if valid_predictions > 0
            else 0.0
        ),
        "expected_train_loss": (
            epoch_loss_numerator / epoch_loss_denominator
            if epoch_loss_denominator > 0.0
            else 0.0
        ),
        "expected_valid_target_count": valid_predictions,
        "tolerance": {"atol": 2.0e-5, "rtol": 2.0e-5},
        "input": tensor_fixture(inputs),
        "targets": tensor_fixture(targets),
        "initial": {
            "weight": tensor_fixture(initial_weight),
            "bias": tensor_fixture(initial_bias),
        },
        "expected_steps": expected_steps,
        "expected": {
            "weight": tensor_fixture(model.weight),
            "bias": tensor_fixture(model.bias),
        },
    }


def gradient_accumulation_matrix() -> list[dict[str, Any]]:
    return [
        gradient_accumulation_case(
            "uneven_microbatch_sample_weighting",
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [0, 1, 1, 0, 1],
            microbatch_size=3,
            grad_accum_steps=2,
        ),
        gradient_accumulation_case(
            "final_partial_window_flush",
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [0, 1, 1, 0, 1, 0, 1],
            microbatch_size=2,
            grad_accum_steps=3,
        ),
        gradient_accumulation_case(
            "weighted_ignored_mean_denominator",
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [0, 1, -100, 1, 0],
            microbatch_size=3,
            grad_accum_steps=2,
            class_weights=[1.0, 4.0],
            ignore_index=-100,
            label_smoothing=0.2,
        ),
        gradient_accumulation_case(
            "weighted_ignored_sum_reduction",
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [0, 1, -100, 1, 0],
            microbatch_size=3,
            grad_accum_steps=2,
            class_weights=[1.0, 4.0],
            ignore_index=-100,
            label_smoothing=0.2,
            loss_reduction="sum",
        ),
    ]


def cross_entropy_case() -> dict[str, Any]:
    logits = torch.tensor(
        [[1.0, 2.0, 0.5], [0.1, -0.2, 0.0]], dtype=torch.float32
    )
    targets = torch.tensor([1, 0], dtype=torch.int64)
    loss = functional.cross_entropy(logits, targets, reduction="mean")
    return {
        "operation": "torch.nn.functional.cross_entropy",
        "dtype": "float32",
        "target_dtype": "int64",
        "reduction": "mean",
        "tolerance": {"atol": 1.0e-5, "rtol": 1.0e-5},
        "logits": tensor_fixture(logits),
        "targets": tensor_fixture(targets),
        "expected": {"loss": float(loss.item())},
    }


def cross_entropy_matrix() -> list[dict[str, Any]]:
    rank2_logits = [
        [2.0, 0.0, -1.0],
        [0.5, 1.5, -0.5],
        [-1.0, 0.2, 2.2],
        [1.0, -0.5, 0.25],
    ]
    rank2_indices = [0, 0, -100, 2]
    rank2_soft = [
        [0.7, 0.2, 0.1],
        [0.0, 1.0, 0.0],
        [0.2, 0.3, 0.5],
        [0.1, 0.2, 0.7],
    ]
    definitions = [
        {
            "name": "index_none",
            "logits": rank2_logits,
            "targets": rank2_indices,
            "target_form": "index",
            "reduction": "none",
        },
        {
            "name": "index_sum_weighted_smoothed_ignored",
            "logits": rank2_logits,
            "targets": rank2_indices,
            "target_form": "index",
            "reduction": "sum",
            "weights": [1.0, 2.0, 4.0],
            "smoothing": 0.2,
        },
        {
            "name": "index_mean_ignored",
            "logits": rank2_logits,
            "targets": rank2_indices,
            "target_form": "index",
            "reduction": "mean",
        },
        {
            "name": "index_mean_weighted_smoothed_ignored",
            "logits": rank2_logits,
            "targets": rank2_indices,
            "target_form": "index",
            "reduction": "mean",
            "weights": [1.0, 2.0, 4.0],
            "smoothing": 0.2,
        },
        {
            "name": "soft_mean_weighted_smoothed",
            "logits": rank2_logits,
            "targets": rank2_soft,
            "target_form": "soft",
            "reduction": "mean",
            "weights": [1.0, 2.0, 4.0],
            "smoothing": 0.2,
        },
        {
            "name": "rank1_index_mean_extreme_logits",
            "logits": [1000.0, -1000.0, 0.0],
            "targets": 1,
            "target_form": "index",
            "reduction": "mean",
            "tolerance": {"atol": 1.0e-4, "rtol": 1.0e-6},
        },
        {
            "name": "rank1_soft_sum_weighted_smoothed",
            "logits": [1.5, -0.25, 0.75],
            "targets": [0.1, 0.6, 0.3],
            "target_form": "soft",
            "reduction": "sum",
            "weights": [1.0, 3.0, 2.0],
            "smoothing": 0.15,
        },
        {
            "name": "rank3_index_none_ignored",
            "logits": [
                [[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]],
                [[-1.0, 0.2, 2.2], [1.0, -0.5, 0.25]],
            ],
            "targets": [[0, -100], [2, 1]],
            "target_form": "index",
            "reduction": "none",
        },
        {
            "name": "rank3_index_mean_weighted_smoothed_ignored",
            "logits": [
                [[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]],
                [[-1.0, 0.2, 2.2], [1.0, -0.5, 0.25]],
            ],
            "targets": [[0, -100], [2, 1]],
            "target_form": "index",
            "reduction": "mean",
            "weights": [1.0, 2.0, 4.0],
            "smoothing": 0.2,
        },
        {
            "name": "rank3_soft_sum_weighted_smoothed",
            "logits": [
                [[1.0, -0.5, 0.25], [0.0, 1.0, -1.0]],
                [[-0.25, 0.5, 1.5], [2.0, 0.0, -0.5]],
            ],
            "targets": [
                [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
                [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]],
            ],
            "target_form": "soft",
            "reduction": "sum",
            "weights": [1.0, 2.0, 4.0],
            "smoothing": 0.1,
        },
        {
            "name": "rank2_index_mean_all_ignored",
            "logits": [[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]],
            "targets": [-100, -100],
            "target_form": "index",
            "reduction": "mean",
            "weights": [1.0, 2.0, 4.0],
            "smoothing": 0.2,
        },
    ]
    cases: list[dict[str, Any]] = []
    for definition in definitions:
        name = definition["name"]
        target_form = definition["target_form"]
        reduction = definition["reduction"]
        weights = definition.get("weights", [])
        smoothing = definition.get("smoothing", 0.0)
        ignore_index = definition.get("ignore_index", -100)
        logits = torch.tensor(
            definition["logits"], dtype=torch.float32, requires_grad=True
        )
        targets = torch.tensor(
            definition["targets"],
            dtype=torch.int64 if target_form == "index" else torch.float32,
        )
        weight = (
            torch.tensor(weights, dtype=torch.float32) if weights else None
        )
        # CyxWiz rank-3 classification tensors are class-last [B, S, C].
        # PyTorch cross_entropy is class-second [B, C, S], so adapt only the
        # independent oracle input and restore gradients to the public shape.
        torch_logits = logits.movedim(-1, 1) if logits.ndim == 3 else logits
        torch_targets = (
            targets.movedim(-1, 1)
            if target_form == "soft" and targets.ndim == 3
            else targets
        )
        loss = functional.cross_entropy(
            torch_logits,
            torch_targets,
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=smoothing,
            reduction=reduction,
        )
        (loss.sum() if reduction == "none" else loss).backward()
        cases.append({
            "name": name,
            "operation": "torch.nn.functional.cross_entropy",
            "dtype": "float32",
            "target_form": target_form,
            "target_dtype": "int64" if target_form == "index" else "float32",
            "reduction": reduction,
            "ignore_index": ignore_index,
            "class_weights": weights,
            "label_smoothing": smoothing,
            "tolerance": definition.get(
                "tolerance", {"atol": 1.0e-5, "rtol": 1.0e-5}
            ),
            "logits": tensor_fixture(logits),
            "targets": tensor_fixture(targets),
            "expected": {
                "loss": (
                    {
                        "shape": list(
                            (loss.reshape(1) if loss.ndim == 0 else loss).shape
                        ),
                        "non_finite": "nan",
                    }
                    if torch.isnan(loss).all()
                    else tensor_fixture(
                        loss.reshape(1) if loss.ndim == 0 else loss
                    )
                ),
                "logit_gradient": tensor_fixture(logits.grad),
            },
        })
    return cases


def adamw_case() -> dict[str, Any]:
    hyperparameters = {
        "learning_rate": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1.0e-8,
        "weight_decay": 0.1,
    }
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    )
    gradient = torch.tensor([0.25, -0.5, 0.125], dtype=torch.float32)
    initial_parameter = parameter.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=hyperparameters["learning_rate"],
        betas=(hyperparameters["beta1"], hyperparameters["beta2"]),
        eps=hyperparameters["epsilon"],
        weight_decay=hyperparameters["weight_decay"],
        foreach=False,
        fused=False,
    )
    parameter.grad = gradient.clone()
    optimizer.step()
    state = optimizer.state[parameter]

    return {
        "operation": "torch.optim.AdamW",
        "dtype": "float32",
        "step_count": 1,
        "hyperparameters": hyperparameters,
        "tolerance": {"atol": 1.0e-5, "rtol": 1.0e-5},
        "initial_parameter": tensor_fixture(initial_parameter),
        "gradient": tensor_fixture(gradient),
        "expected": {
            "parameter": tensor_fixture(parameter),
            "exp_avg": tensor_fixture(state["exp_avg"]),
            "exp_avg_sq": tensor_fixture(state["exp_avg_sq"]),
            "step": int(state["step"].item()),
        },
    }


def weighted_sampler_case() -> dict[str, Any]:
    class_counts = [3072, 1024]
    labels = torch.repeat_interleave(
        torch.arange(len(class_counts), dtype=torch.int64),
        torch.tensor(class_counts, dtype=torch.int64),
    )
    class_count_tensor = torch.tensor(class_counts, dtype=torch.float64)
    sample_weights = class_count_tensor.reciprocal()[labels]
    expected_probabilities = torch.stack(
        [sample_weights[labels == label].sum()
         for label in range(len(class_counts))]
    )
    expected_probabilities /= expected_probabilities.sum()

    seed = 3901
    num_samples = sum(class_counts)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
    sampled_indices = torch.tensor(list(sampler), dtype=torch.int64)
    empirical_counts = torch.bincount(
        labels[sampled_indices], minlength=len(class_counts)
    )

    return {
        "operation": "torch.utils.data.WeightedRandomSampler",
        "class_counts": class_counts,
        "sample_weight_rule": "inverse_class_frequency",
        "replacement": True,
        "num_samples": num_samples,
        "seed": seed,
        "expected_class_probabilities": expected_probabilities.tolist(),
        "pytorch_empirical_class_counts": empirical_counts.tolist(),
        "absolute_probability_tolerance": 0.03,
        "cross_rng_probability_tolerance": 0.04,
        "rng_contract": (
            "Distribution parity is required; the C++ RNG stream is not "
            "required to reproduce torch.multinomial indices."
        ),
    }


def scheduler_lr_case(
    name: str,
    scheduler_type: str,
    scheduler_parameters: dict[str, Any],
    steps: int,
    metrics: list[float] | None = None,
) -> dict[str, Any]:
    base_lr = 0.1
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=base_lr)

    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_parameters["step_size"],
            gamma=scheduler_parameters["gamma"],
        )
    elif scheduler_type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=scheduler_parameters["gamma"]
        )
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_parameters["T_max"],
            eta_min=scheduler_parameters["eta_min"],
        )
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_parameters["mode"],
            factor=scheduler_parameters["factor"],
            patience=scheduler_parameters["patience"],
            threshold=scheduler_parameters["threshold"],
            threshold_mode="abs",
            min_lr=scheduler_parameters["min_lr"],
        )
    elif scheduler_type == "LinearWarmupLR":
        warmup_epochs = scheduler_parameters["warmup_epochs"]
        start_lr = scheduler_parameters["start_lr"]
        start_factor = start_lr / base_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: start_factor
            + (1.0 - start_factor) * min(epoch / warmup_epochs, 1.0),
        )
    elif scheduler_type == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_parameters["max_lr"],
            total_steps=scheduler_parameters["total_steps"],
            pct_start=scheduler_parameters["pct_start"],
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=scheduler_parameters["div_factor"],
            final_div_factor=scheduler_parameters["final_div_factor"],
            three_phase=False,
        )
    else:
        raise ValueError(f"unsupported scheduler fixture type: {scheduler_type}")

    initial_learning_rate = optimizer.param_groups[0]["lr"]
    expected_steps: list[dict[str, Any]] = []
    for index in range(1, steps + 1):
        optimizer.step()
        expected: dict[str, Any] = {"index": index}
        if metrics is None:
            scheduler.step()
        else:
            metric = metrics[index - 1]
            scheduler.step(metric)
            expected["metric"] = metric
        expected["learning_rate"] = optimizer.param_groups[0]["lr"]
        expected_steps.append(expected)

    result = {
        "name": name,
        "operation": (
            "torch.optim.lr_scheduler.LambdaLR(linear_warmup)"
            if scheduler_type == "LinearWarmupLR"
            else f"torch.optim.lr_scheduler.{scheduler_type}"
        ),
        "call_order": "optimizer.step_then_scheduler.step",
        "base_learning_rate": base_lr,
        "parameters": scheduler_parameters,
        "tolerance": {"atol": 1.0e-12, "rtol": 1.0e-12},
        "expected_initial_learning_rate": initial_learning_rate,
        "expected_steps": expected_steps,
        "expected_reset_learning_rate": base_lr
        if scheduler_type not in {"LinearWarmupLR", "OneCycleLR"}
        else (
            start_lr
            if scheduler_type == "LinearWarmupLR"
            else scheduler_parameters["max_lr"] / scheduler_parameters["div_factor"]
        ),
    }
    if scheduler_type == "OneCycleLR":
        result["expected_error_step"] = scheduler_parameters["total_steps"] + 1
    return result


def scheduler_lr_matrix() -> list[dict[str, Any]]:
    return [
        scheduler_lr_case(
            "step_lr_epoch_sequence",
            "StepLR",
            {"step_size": 2, "gamma": 0.5},
            steps=5,
        ),
        scheduler_lr_case(
            "exponential_lr_epoch_sequence",
            "ExponentialLR",
            {"gamma": 0.9},
            steps=4,
        ),
        scheduler_lr_case(
            "cosine_annealing_boundary_sequence",
            "CosineAnnealingLR",
            {"T_max": 4, "eta_min": 0.01},
            steps=5,
        ),
        scheduler_lr_case(
            "plateau_patience_sequence",
            "ReduceLROnPlateau",
            {
                "mode": "min",
                "factor": 0.5,
                "patience": 2,
                "threshold": 1.0e-4,
                "min_lr": 0.01,
            },
            steps=5,
            metrics=[1.0, 1.1, 1.2, 1.3, 1.4],
        ),
        scheduler_lr_case(
            "plateau_max_threshold_minimum_sequence",
            "ReduceLROnPlateau",
            {
                "mode": "max",
                "factor": 0.5,
                "patience": 1,
                "threshold": 0.05,
                "min_lr": 0.01,
            },
            steps=13,
            metrics=[
                0.50,
                0.54,
                0.56,
                0.55,
                0.54,
                0.53,
                0.52,
                0.51,
                0.50,
                0.49,
                0.48,
                0.47,
                0.46,
            ],
        ),
        scheduler_lr_case(
            "linear_warmup_boundary_sequence",
            "LinearWarmupLR",
            {"warmup_epochs": 4, "start_lr": 0.0},
            steps=6,
        ),
        scheduler_lr_case(
            "one_cycle_two_phase_cosine_sequence",
            "OneCycleLR",
            {
                "max_lr": 0.1,
                "total_steps": 10,
                "pct_start": 0.3,
                "div_factor": 25.0,
                "final_div_factor": 10000.0,
                "anneal_strategy": "cos",
                "cycle_momentum": False,
                "three_phase": False,
            },
            steps=10,
        ),
    ]


def optimizer_lr_warmup_case(
    name: str,
    warmup_type: str,
    warmup_steps: int = 4,
    steps: int = 6,
) -> dict[str, Any]:
    base_lr = 0.1
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=base_lr)

    if warmup_type == "linear":
        lr_lambda = lambda step: min(step / warmup_steps, 1.0)
    elif warmup_type == "cosine":
        lr_lambda = lambda step: 0.5 * (
            1.0 - math.cos(math.pi * min(step / warmup_steps, 1.0))
        )
    elif warmup_type == "none":
        lr_lambda = lambda step: 1.0
    else:
        raise ValueError(f"unsupported optimizer warmup type: {warmup_type}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda
    )
    expected_steps: list[dict[str, Any]] = []
    for index in range(1, steps + 1):
        optimizer.zero_grad()
        parameter.grad = torch.tensor([1.0], dtype=torch.float32)
        optimizer.step()
        scheduler.step()
        expected_steps.append(
            {
                "index": index,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "parameter": parameter.detach().item(),
                "warmup_progress": (
                    1.0
                    if warmup_type == "none"
                    else min(index / warmup_steps, 1.0)
                ),
                "warmup_complete": (
                    warmup_type == "none" or index >= warmup_steps
                ),
            }
        )

    return {
        "name": name,
        "operation": "torch.optim.SGD + torch.optim.lr_scheduler.LambdaLR",
        "call_order": "optimizer.step_then_scheduler.step",
        "base_learning_rate": base_lr,
        "warmup_type": warmup_type,
        "warmup_steps": warmup_steps,
        "initial_parameter": 1.0,
        "gradient": 1.0,
        "expected_initial_learning_rate": (
            0.0 if warmup_type != "none" else base_lr
        ),
        "expected_steps": expected_steps,
        "tolerance": {"atol": 1.0e-7, "rtol": 1.0e-7},
    }


def optimizer_lr_warmup_matrix() -> list[dict[str, Any]]:
    return [
        optimizer_lr_warmup_case("optimizer_linear_warmup", "linear"),
        optimizer_lr_warmup_case("optimizer_cosine_warmup", "cosine"),
        optimizer_lr_warmup_case("optimizer_no_warmup", "none"),
    ]


def generate_fixture() -> dict[str, Any]:
    torch.manual_seed(39)
    torch.use_deterministic_algorithms(True)
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle": {
            "name": "PyTorch",
            "version": torch.__version__,
            "device": "cpu",
        },
        "seed": 39,
        "cases": {
            "dropout_semantics_f32": dropout_semantics(),
            "flatten_forward_backward_f32": flatten_matrix(),
            "tensor_concat_f32": tensor_concat_matrix(),
            "tensor_elementwise": tensor_elementwise_matrix(),
            "tensor_indexing_f32": tensor_indexing_matrix(),
            "tensor_permute_f32": tensor_permute_matrix(),
            "tensor_reductions": tensor_reduction_matrix(),
            "tensor_shape_semantics_f32": tensor_shape_semantics(),
            "linear_basic_f32": linear_case(),
            "cross_entropy_index_mean_f32": cross_entropy_case(),
            "cross_entropy_matrix_f32": cross_entropy_matrix(),
            "adamw_step1_f32": adamw_case(),
            "gradient_accumulation_linear_ce_sgd_f32":
                gradient_accumulation_matrix(),
            "weighted_sampler_inverse_frequency": weighted_sampler_case(),
            "scheduler_lr_sequences": scheduler_lr_matrix(),
            "optimizer_lr_warmup_sequences": optimizer_lr_warmup_matrix(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(generate_fixture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
