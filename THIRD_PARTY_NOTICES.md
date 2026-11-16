# Third-party notices

## External controller and robot resources

SONIC is an external dependency from
https://github.com/NVlabs/GR00T-WholeBodyControl. The validated upstream revision
is recorded in `scripts/setup_sonic.py`. Its source, model weights, G1 assets,
and Git history are not included in this project's source tree.

The decoder adapter reproduces the upstream release observation layout,
joint-order mapping, default joint positions, action scales, and PD constants
from `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`.
The collection adapter follows the upstream planner's reference resampling and
crossfade conventions. These backend-specific details are distinct from this
project's path tracking, stopping controller, collection, and reporting tools.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Upstream source code is distributed under Apache License 2.0; upstream model
weights are covered by the NVIDIA Open Model License. The original upstream
license notice is retained at [licenses/SONIC-LICENSE.txt](licenses/SONIC-LICENSE.txt).
Downloaded dependency files retain their own upstream notices and terms.
