#!/usr/bin/env python3
"""torchvision fixtures for the image transform nodes (TOFIX140 Group C).

The Engine's image batch is rows [N, H*W*C], each row an [H, W, C] image in
[0, 1]; torchvision works on [N, C, H, W]. Each case applies one transform
with fixed, per-sample settings (the values a random node would draw: crop
position, flip yes/no, angle, jitter factors and order) through
torchvision.transforms.functional, and stores the input and output rows.

    PYTHONPATH=<dir with torchvision> py -3.12 generate_image_transform_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as F2
from torchvision.transforms import InterpolationMode

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "image_transforms_torchvision.json"


def rows(images: torch.Tensor) -> dict[str, Any]:
    """[N, C, H, W] -> the Engine's rows [N, H*W*C] (HWC order)."""
    n, c, h, w = images.shape
    return {"shape": [n, h, w, c], "values": images.permute(0, 2, 3, 1).reshape(-1).tolist()}


def batch(seed: int, n: int, c: int, h: int, w: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((n, c, h, w), generator=generator, dtype=torch.float64).float()


JITTER = [
    lambda img, f: F.adjust_brightness(img, f),
    lambda img, f: F.adjust_contrast(img, f),
    lambda img, f: F.adjust_saturation(img, f),
    lambda img, f: F.adjust_hue(img, f),
]


def per_sample(images: torch.Tensor, fn: Callable[[torch.Tensor, int], torch.Tensor]) -> torch.Tensor:
    return torch.stack([fn(images[i], i) for i in range(images.shape[0])])


def case(name: str, op: dict[str, Any], draws: dict[str, Any], images: torch.Tensor,
         out: torch.Tensor) -> dict[str, Any]:
    return {"name": name, "op": op, "draws": draws, "input": rows(images), "expected": rows(out)}


def mix_cases() -> list[dict[str, Any]]:
    """torchvision.transforms.v2 MixUp / CutMix run for real; the lambda and box
    they drew are recorded so the Engine can replay the same mix."""
    images = batch(143, 4, 3, 6, 7)
    labels = torch.tensor([0, 2, 1, 2])
    cases = []
    for method, transform, seed in (("mixup", v2.MixUp(alpha=1.0, num_classes=3), 7),
                                    ("cutmix", v2.CutMix(alpha=1.0, num_classes=3), 11),
                                    ("cutmix", v2.CutMix(alpha=0.4, num_classes=3), 23)):
        torch.manual_seed(seed)
        params = transform.make_params([images])
        torch.manual_seed(seed)
        mixed, mixed_labels = transform(images, labels)
        draws: dict[str, Any] = {}
        if method == "mixup":
            draws["lambda"] = params["lam"]
        else:
            x1, y1, x2, y2 = params["box"]
            draws.update({"lambda": params["lam_adjusted"], "top": y1, "left": x1,
                          "height": y2 - y1, "width": x2 - x1})
        cases.append({
            "name": f"{method}_seed{seed}", "op": {"kind": "mix", "method": method}, "draws": draws,
            "input": rows(images), "expected": rows(mixed),
            "labels": {"shape": [4, 3], "values": torch.nn.functional.one_hot(labels, 3).float().reshape(-1).tolist()},
            "expected_labels": {"shape": [4, 3], "values": mixed_labels.reshape(-1).tolist()},
        })
    return cases


RANDAUGMENT_OPS = ["Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate", "Brightness", "Color",
                   "Contrast", "Sharpness", "Posterize", "Solarize", "AutoContrast", "Equalize"]


def randaugment_cases() -> list[dict[str, Any]]:
    """Each RandAugment op through torchvision v2's own dispatcher
    (_apply_image_or_video_transform) with fixed signed magnitudes, plus a
    two-pick sequence."""
    ra = v2.RandAugment()
    images = batch(144, 2, 3, 9, 8)

    def apply(img: torch.Tensor, ops: list[int], mags: list[float]) -> torch.Tensor:
        for op, m in zip(ops, mags):
            img = ra._apply_image_or_video_transform(img, RANDAUGMENT_OPS[op], m,
                                                     interpolation=ra.interpolation, fill=ra._fill)
        return img

    def magnitude(op: int, bin_: int, h: int, w: int) -> float:
        fn, _ = ra._AUGMENTATION_SPACE[RANDAUGMENT_OPS[op]]
        values = fn(31, h, w)
        return 0.0 if values is None else float(values[bin_])

    cases = []
    for op in range(1, 14):
        mags = [magnitude(op, 9, 9, 8), magnitude(op, 20, 9, 8)]
        if ra._AUGMENTATION_SPACE[RANDAUGMENT_OPS[op]][1]:
            mags[1] = -mags[1]
        order = [[op], [op]]
        factors = [[mags[0]], [mags[1]]]
        cases.append(case(f"randaugment_{RANDAUGMENT_OPS[op].lower()}", {"kind": "randaugment"},
                          {"order": order, "factors": factors}, images,
                          per_sample(images, lambda img, i: apply(img, order[i], factors[i]))))
    gray = batch(145, 2, 1, 9, 8)
    for op in (9, 12, 13):  # sharpness, autocontrast, equalize on one channel
        order = [[op], [op]]
        factors = [[magnitude(op, 9, 9, 8)], [magnitude(op, 25, 9, 8)]]
        cases.append(case(f"randaugment_{RANDAUGMENT_OPS[op].lower()}_one_channel", {"kind": "randaugment"},
                          {"order": order, "factors": factors}, gray,
                          per_sample(gray, lambda img, i: apply(img, order[i], factors[i]))))
    order = [[1, 13], [10, 5]]
    factors = [[-magnitude(1, 12, 9, 8), 0.0], [magnitude(10, 9, 9, 8), magnitude(5, 9, 9, 8)]]
    cases.append(case("randaugment_two_picks", {"kind": "randaugment"}, {"order": order, "factors": factors},
                      images, per_sample(images, lambda img, i: apply(img, order[i], factors[i]))))
    return cases


AUGMENT_OPS = RANDAUGMENT_OPS + ["Invert"]


def recorded(transform: Any, image: torch.Tensor, seed: int) -> tuple[torch.Tensor, list[tuple[Any, str, float]]]:
    """The transform class run for real on one [C, H, W] image under a seed,
    with each op it applied recorded as (its input, name, magnitude)."""
    calls: list[tuple[Any, str, float]] = []
    apply = transform._apply_image_or_video_transform

    def logging_apply(img, transform_id, magnitude, interpolation, fill):
        calls.append((img, transform_id, magnitude))
        return apply(img, transform_id, magnitude, interpolation=interpolation, fill=fill)

    transform._apply_image_or_video_transform = logging_apply
    torch.manual_seed(seed)
    out = transform(image)
    return out, calls


def policy_cases() -> list[dict[str, Any]]:
    """AutoAugment, TrivialAugmentWide and AugMix run for real (seeded) on
    each sample; the ops, magnitudes and mixing weights they drew are recorded
    so the Engine replays them."""
    images = batch(146, 2, 3, 9, 8)
    gray = batch(147, 2, 1, 9, 8)
    cases = []

    def index(name: str) -> int:
        return AUGMENT_OPS.index(name)

    for policy, label, seeds in ((v2.AutoAugmentPolicy.IMAGENET, "imagenet", (3, 8)),
                                 (v2.AutoAugmentPolicy.CIFAR10, "cifar10", (4, 12)),
                                 (v2.AutoAugmentPolicy.SVHN, "svhn", (1, 6))):
        outs, order, factors = [], [], []
        for i, seed in enumerate(seeds):
            out, calls = recorded(v2.AutoAugment(policy), images[i], seed)
            outs.append(out)
            ops = [index(name) for _, name, _ in calls] + [0] * (2 - len(calls))
            mags = [m for _, _, m in calls] + [0.0] * (2 - len(calls))
            order.append(ops)
            factors.append(mags)
        cases.append(case(f"autoaugment_{label}", {"kind": "autoaugment"}, {"order": order, "factors": factors},
                          images, torch.stack(outs)))
    # Invert, AutoAugment's own op, through the same dispatcher.
    aa = v2.AutoAugment()
    cases.append(case("autoaugment_invert", {"kind": "autoaugment"}, {"order": [[14, 0], [14, 0]],
                      "factors": [[0.0, 0.0], [0.0, 0.0]]}, gray,
                      per_sample(gray, lambda img, i: aa._apply_image_or_video_transform(
                          img, "Invert", 0.0, interpolation=aa.interpolation, fill=aa._fill))))

    outs, order, factors = [], [], []
    for i, seed in enumerate((5, 9)):
        out, calls = recorded(v2.TrivialAugmentWide(), images[i], seed)
        outs.append(out)
        order.append([index(calls[0][1])])
        factors.append([calls[0][2]])
    cases.append(case("trivialaugment_seeded", {"kind": "trivialaugment"}, {"order": order, "factors": factors},
                      images, torch.stack(outs)))
    # Every Wide op at a large bin (signed ones negative on the second sample).
    ta = v2.TrivialAugmentWide()
    for op in range(1, 14):
        name = RANDAUGMENT_OPS[op]
        fn, signed = ta._AUGMENTATION_SPACE[name]
        values = fn(31, 9, 8)
        m = 0.0 if values is None else float(values[27])
        mags = [m, -m if signed else m]
        cases.append(case(f"trivialaugment_{name.lower()}", {"kind": "trivialaugment"},
                          {"order": [[op], [op]], "factors": [[mags[0]], [mags[1]]]}, images,
                          per_sample(images, lambda img, i: ta._apply_image_or_video_transform(
                              img, name, mags[i], interpolation=ta.interpolation, fill=ta._fill))))

    for label, kwargs, seeds, source in (("default", {}, (2, 7), images),
                                         ("depth2_width4", {"chain_depth": 2, "mixture_width": 4, "severity": 6}, (3, 10), images),
                                         ("one_channel", {"alpha": 0.5}, (4, 11), gray)):
        transform = v2.AugMix(**kwargs)
        width = transform.mixture_width
        slots = transform.chain_depth if transform.chain_depth > 0 else 3
        dirichlet = transform._sample_dirichlet
        outs, order, factors, weights = [], [], [], []
        for i, seed in enumerate(seeds):
            drawn: list[torch.Tensor] = []

            def recording_dirichlet(params, drawn=drawn):
                value = dirichlet(params)
                drawn.append(value)
                return value

            transform._sample_dirichlet = recording_dirichlet
            out, calls = recorded(transform, source[i], seed)
            outs.append(out)
            m, mixing = drawn
            weights.append([float(m[0, 0])] + (mixing * m[:, 1].reshape(1, -1)).reshape(-1).tolist())
            # Each chain starts again from the same batch object.
            chains: list[list[tuple[str, float]]] = []
            for img, name, magnitude in calls:
                if img is calls[0][0]:
                    chains.append([])
                chains[-1].append((name, magnitude))
            assert len(chains) == width, (label, len(chains))
            ops, mags = [], []
            for chain in chains:
                ops += [index(name) for name, _ in chain] + [0] * (slots - len(chain))
                mags += [mag for _, mag in chain] + [0.0] * (slots - len(chain))
            order.append(ops)
            factors.append(mags)
        spec = {"kind": "augmix", "mixture_width": width, "interpolation": "bilinear"}
        cases.append(case(f"augmix_{label}", spec, {"order": order, "factors": factors, "weights": weights},
                          source, torch.stack(outs)))
    return cases


def geometry_cases() -> list[dict[str, Any]]:
    """Random Resized Crop, Random Affine and Random Perspective with the
    params their transform classes draw (seeded), applied by those classes."""
    images = batch(148, 2, 3, 9, 8)
    cases = []

    def run(transform: Any, seeds: tuple[int, ...], source: torch.Tensor) -> tuple[list[Any], torch.Tensor]:
        params: list[Any] = []
        make = transform.make_params

        def recording_make(flat_inputs):
            value = make(flat_inputs)
            params.append(value)
            return value

        transform.make_params = recording_make
        outs = []
        for i, seed in enumerate(seeds):
            before = len(params)
            torch.manual_seed(seed)
            outs.append(transform(source[i]))
            if len(params) == before:
                params.append(None)  # not applied (the class's probability)
        return params, torch.stack(outs)

    for interpolation, mode in (("bilinear", InterpolationMode.BILINEAR), ("nearest", InterpolationMode.NEAREST)):
        for size in ((5, 6), (12, 10)):
            params, out = run(v2.RandomResizedCrop(size, interpolation=mode), (13, 21), images)
            cases.append(case(f"resized_crop_{interpolation}_{size[0]}x{size[1]}",
                              {"kind": "random_resized_crop", "height": size[0], "width": size[1],
                               "interpolation": interpolation},
                              {"top": [p["top"] for p in params], "left": [p["left"] for p in params],
                               "box_height": [p["height"] for p in params], "box_width": [p["width"] for p in params]},
                              images, out))
    # The box the same size as the output: the crop itself.
    cases.append(case("resized_crop_same_size", {"kind": "random_resized_crop", "height": 4, "width": 3,
                                                 "interpolation": "bilinear"},
                      {"top": [2, 0], "left": [1, 5], "box_height": [4, 4], "box_width": [3, 3]}, images,
                      per_sample(images, lambda img, i: F2.crop(img, [2, 0][i], [1, 5][i], 4, 3))))

    for interpolation, mode in (("nearest", InterpolationMode.NEAREST), ("bilinear", InterpolationMode.BILINEAR)):
        for fill in (0.0, 0.5):
            transform = v2.RandomAffine(degrees=25, translate=(0.2, 0.3), scale=(0.8, 1.2), shear=(-10, 10, -5, 8),
                                        interpolation=mode, fill=fill)
            params, out = run(transform, (31, 32), images)
            factors = [[p["angle"], float(p["translate"][0]), float(p["translate"][1]), p["scale"],
                        p["shear"][0], p["shear"][1]] for p in params]
            cases.append(case(f"affine_{interpolation}_fill{fill}",
                              {"kind": "random_affine", "interpolation": interpolation, "fill": fill},
                              {"factors": factors}, images, out))

    import torchvision.transforms.v2._geometry as geometry
    coeffs = geometry._get_perspective_coeffs
    corners: list[list[int]] = []

    def recording_coeffs(startpoints, endpoints):
        corners.append([v for point in endpoints for v in point])
        return coeffs(startpoints, endpoints)

    geometry._get_perspective_coeffs = recording_coeffs
    for interpolation, mode, fill, seeds in (("bilinear", InterpolationMode.BILINEAR, 0.0, (40, 41)),
                                             ("nearest", InterpolationMode.NEAREST, 0.25, (42, 43))):
        corners.clear()
        params, out = run(v2.RandomPerspective(distortion_scale=0.6, p=1.0, interpolation=mode, fill=fill),
                          seeds, images)
        cases.append(case(f"perspective_{interpolation}_fill{fill}",
                          {"kind": "random_perspective", "interpolation": interpolation, "fill": fill},
                          {"apply": [1, 1], "corners": [list(c) for c in corners]}, images, out))
    geometry._get_perspective_coeffs = coeffs
    # Not applied: the sample passes through.
    cases.append(case("perspective_not_applied", {"kind": "random_perspective", "interpolation": "bilinear"},
                      {"apply": [0, 0], "corners": [[0] * 8, [0] * 8]}, images, images))
    return cases


def noise_and_gray_cases() -> list[dict[str, Any]]:
    images = batch(149, 2, 3, 6, 7)
    gray = batch(150, 2, 1, 6, 7)
    cases = []
    for label, mean, sigma, clip, seed in (("clipped", 0.0, 0.1, True, 51), ("unclipped", 0.05, 0.3, False, 52)):
        torch.manual_seed(seed)
        out = v2.GaussianNoise(mean=mean, sigma=sigma, clip=clip)(images)
        torch.manual_seed(seed)
        noise = torch.randn_like(images)
        cases.append(case(f"gaussian_noise_{label}",
                          {"kind": "gaussian_noise", "noise_mean": mean, "sigma": sigma, "clip": clip},
                          {"apply": [1, 1], "noise": rows(noise)["values"]}, images, out))
    apply = [1, 0]
    cases.append(case("random_grayscale", {"kind": "random_grayscale"}, {"apply": apply}, images,
                      per_sample(images, lambda img, i: F2.rgb_to_grayscale(img, num_output_channels=3)
                                 if apply[i] else img)))
    cases.append(case("random_grayscale_one_channel", {"kind": "random_grayscale"}, {"apply": [1, 1]}, gray,
                      per_sample(gray, lambda img, i: F2.rgb_to_grayscale(img, num_output_channels=1))))
    return cases


def magnitude_tables() -> list[dict[str, Any]]:
    """Each policy's magnitude table, as torchvision computes it."""
    spaces = (("standard", 31, v2.RandAugment._AUGMENTATION_SPACE),
              ("standard", 10, v2.AutoAugment._AUGMENTATION_SPACE),
              ("wide", 31, v2.TrivialAugmentWide._AUGMENTATION_SPACE),
              ("wide", 8, v2.TrivialAugmentWide._AUGMENTATION_SPACE),
              ("augmix", 10, v2.AugMix._AUGMENTATION_SPACE))
    tables = []
    for space, bins, ops in spaces:
        for h, w in ((9, 8), (224, 224), (32, 40)):
            table = {}
            for name, (fn, signed) in ops.items():
                values = fn(bins, h, w)
                table[name] = {"signed": signed,
                               "values": [0.0] * bins if values is None else [float(v) for v in values]}
            tables.append({"space": space, "bins": bins, "height": h, "width": w, "ops": table})
    return tables


def build() -> list[dict[str, Any]]:
    rgb = batch(140, 2, 3, 6, 7)
    gray = batch(141, 2, 1, 6, 7)
    big = batch(142, 2, 3, 9, 8)
    cases = []

    cases.append(case("center_crop_4x5", {"kind": "center_crop", "height": 4, "width": 5}, {},
                      rgb, F.center_crop(rgb, [4, 5])))
    # (6 - 3) / 2 = 1.5 and (7 - 4) / 2 = 1.5: Python rounds half to even -> 2, 2.
    cases.append(case("center_crop_tie_3x4", {"kind": "center_crop", "height": 3, "width": 4}, {},
                      rgb, F.center_crop(rgb, [3, 4])))

    tops, lefts = [0, 2], [2, 0]
    cases.append(case("random_crop_positions", {"kind": "random_crop", "height": 4, "width": 5},
                      {"top": tops, "left": lefts}, rgb,
                      per_sample(rgb, lambda img, i: F.crop(img, tops[i], lefts[i], 4, 5))))

    # torchvision RandomCrop(padding=2): zero border, then the crop at (top, left) of the padded image.
    ptops, plefts = [0, 4], [3, 1]
    padded = F.pad(rgb, [2, 2, 2, 2], fill=0)
    cases.append(case("random_crop_padding", {"kind": "random_crop", "height": 6, "width": 7, "padding": 2},
                      {"top": ptops, "left": plefts}, rgb,
                      per_sample(padded, lambda img, i: F.crop(img, ptops[i], plefts[i], 6, 7))))

    apply = [1, 0]
    cases.append(case("horizontal_flip", {"kind": "horizontal_flip"}, {"apply": apply}, rgb,
                      per_sample(rgb, lambda img, i: F.hflip(img) if apply[i] else img)))
    cases.append(case("vertical_flip", {"kind": "vertical_flip"}, {"apply": [0, 1]}, rgb,
                      per_sample(rgb, lambda img, i: F.vflip(img) if i == 1 else img)))

    # RandomRotation's fill=0: outside pixels through the fill mask (it changes bilinear borders).
    for interpolation, mode in (("nearest", InterpolationMode.NEAREST), ("bilinear", InterpolationMode.BILINEAR)):
        for label, angles, images in (("odd", [30.0, -45.0], rgb), ("even", [90.0, 17.5], big)):
            cases.append(case(f"rotate_{interpolation}_{label}",
                              {"kind": "rotate", "interpolation": interpolation},
                              {"angle": angles}, images,
                              per_sample(images, lambda img, i: F2.rotate(img, angles[i], interpolation=mode, fill=0))))

    for index, (label, factors) in enumerate((("brightness", [0.5, 1.4]), ("contrast", [0.6, 1.3]),
                                               ("saturation", [0.2, 1.7]), ("hue", [-0.3, 0.25]))):
        neutral = [1.0, 1.0, 1.0, 0.0]
        jitter = [[*neutral] for _ in factors]
        for i, factor in enumerate(factors):
            jitter[i][index] = factor
        cases.append(case(f"jitter_{label}", {"kind": "color_jitter"},
                          {"factors": jitter, "order": [[index]] * 2}, rgb,
                          per_sample(rgb, lambda img, i: JITTER[index](img, factors[i]))))

    factors = [[1.2, 0.8, 1.5, 0.1], [0.7, 1.2, 0.5, -0.2]]
    orders = [[2, 0, 3, 1], [0, 1, 2, 3]]

    def jitter_all(img: torch.Tensor, i: int) -> torch.Tensor:
        for op in orders[i]:
            img = JITTER[op](img, factors[i][op])
        return img

    cases.append(case("jitter_all_orders", {"kind": "color_jitter"},
                      {"factors": factors, "order": orders}, rgb, per_sample(rgb, jitter_all)))
    gray_factors = [[0.6, 1.4, 1.5, 0.2], [1.3, 0.5, 0.4, -0.1]]
    gray_orders = [[3, 2, 1, 0], [1, 0, 2, 3]]

    def jitter_gray(img: torch.Tensor, i: int) -> torch.Tensor:
        for op in gray_orders[i]:
            img = JITTER[op](img, gray_factors[i][op])
        return img

    cases.append(case("jitter_one_channel", {"kind": "color_jitter"},
                      {"factors": gray_factors, "order": gray_orders}, gray, per_sample(gray, jitter_gray)))

    for k, sigma in ((3, 0.8), (5, 1.5)):
        cases.append(case(f"gaussian_blur_k{k}", {"kind": "gaussian_blur", "kernel_size": k, "sigma": sigma}, {},
                          rgb, F.gaussian_blur(rgb, [k, k], [sigma, sigma])))
    cases.append(case("gaussian_blur_one_channel", {"kind": "gaussian_blur", "kernel_size": 3, "sigma": 1.1}, {},
                      gray, F.gaussian_blur(gray, [3, 3], [1.1, 1.1])))

    cases.append(case("grayscale", {"kind": "grayscale"}, {}, rgb, F.rgb_to_grayscale(rgb)))
    cases.extend(randaugment_cases())
    cases.extend(policy_cases())
    cases.extend(geometry_cases())
    cases.extend(noise_and_gray_cases())

    # Advanced Augment erasing (Cutout / Random Erasing): F.erase on each sample's drawn box;
    # a 0-sized box leaves the sample unchanged.
    boxes = [(1, 2, 3, 4), (0, 0, 0, 0)]
    cases.append(case("erase_boxes_value0", {"kind": "erase", "value": 0.0},
                      {"top": [b[0] for b in boxes], "left": [b[1] for b in boxes],
                       "box_height": [b[2] for b in boxes], "box_width": [b[3] for b in boxes]}, rgb,
                      per_sample(rgb, lambda img, i: F.erase(img, *boxes[i], torch.tensor(0.0)) if boxes[i][2] else img)))
    clipped = [(0, 5, 3, 2), (4, 0, 2, 3)]  # Cutout squares clipped at the image edge
    cases.append(case("erase_clipped_value_half", {"kind": "erase", "value": 0.5},
                      {"top": [b[0] for b in clipped], "left": [b[1] for b in clipped],
                       "box_height": [b[2] for b in clipped], "box_width": [b[3] for b in clipped]}, gray,
                      per_sample(gray, lambda img, i: F.erase(img, *clipped[i], torch.tensor(0.5)))))

    # Morphology with a flat square: max_pool2d's implicit -inf border ignores outside pixels
    # (kornia.morphology's geodesic border, OpenCV's default morphology border).
    def dilate(img: torch.Tensor, k: int) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(img, k, stride=1, padding=k // 2)

    def erode(img: torch.Tensor, k: int) -> torch.Tensor:
        return -dilate(-img, k)

    morph = {
        "erode": lambda x, k: erode(x, k),
        "dilate": lambda x, k: dilate(x, k),
        "open": lambda x, k: dilate(erode(x, k), k),
        "close": lambda x, k: erode(dilate(x, k), k),
        "gradient": lambda x, k: dilate(x, k) - erode(x, k),
        "tophat": lambda x, k: x - dilate(erode(x, k), k),
        "blackhat": lambda x, k: erode(dilate(x, k), k) - x,
    }
    for operation, fn in morph.items():
        cases.append(case(f"morphology_{operation}", {"kind": "morphology", "operation": operation, "kernel_size": 3},
                          {}, rgb, fn(rgb, 3)))
    cases.append(case("morphology_erode_k5_one_channel",
                      {"kind": "morphology", "operation": "erode", "kernel_size": 5}, {}, gray, erode(gray, 5)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": "generate_image_transform_fixtures.py",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cases": build(),
        "mix_cases": mix_cases(),
        "augment_magnitudes": magnitude_tables(),
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
