# Licensing and Commercial Distribution Policy

## This is engineering policy, not legal advice

Product counsel must approve each distributed runtime and each catalogued model
before release. Licenses for framework code, runtime wheels, CUDA-related
components, model source code, and model weights are independent.

## Framework code

PyTorch is BSD-3-Clause; JAX and Flax are Apache-2.0. These permissive licenses
can generally coexist with proprietary commercial software, subject to their
notice, attribution, and redistribution conditions. They do not automatically
cover every dependency bundled in a runtime, nor any model weights loaded by
that runtime.

For every shipped runtime, generate an SBOM and third-party notice bundle from
the exact lockfile/wheels. Review GPU/runtime redistribution terms separately.

## Model code versus model weights

Model source and weights often have different licenses. The TabFM repository
source is Apache-2.0, but its published model card states that the weights use
the TabFM Non-Commercial License v1.0. Therefore TabFM weights are not an
approved default for a paid CyxWiz product, marketplace, hosted inference, or
commercial deployment until legal approval or a commercial grant exists.

## Catalog policy

Each model entry needs verified fields:

```text
repository, immutable revision, source license, weight license,
commercial_use = allowed|restricted|unknown,
redistribution = allowed|restricted|unknown,
attribution, terms URL, legal approval record, review date
```

Only `commercial_use=allowed` and `redistribution=allowed` models may be
bundled or offered in a commercial default catalog. Restricted models can at
most be separated into a clearly labelled, legally approved user-download flow;
do not make that decision solely in code.

## Product naming

Describe integrations factually (for example, "runs with PyTorch") but do not
imply endorsement by framework or model owners. Respect applicable trademarks.

## Operational controls

- Do not ship access tokens.
- Do not upload user data to a model host without explicit product consent.
- Preserve required notices with distributed runtime artifacts.
- Block marketplace publication when license metadata is absent or unapproved.
- Re-review licenses when runtime/model revisions change.

