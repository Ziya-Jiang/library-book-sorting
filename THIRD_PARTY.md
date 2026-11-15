# Third-party components

This private research snapshot does not grant a new license to third-party code,
hardware CAD models or vendor binaries. Existing copyright and license notices
remain in their source directories. No license is assigned to the project by
this export.

- `third_party/curobo`: NVIDIA CuRobo, upstream commit and local patch recorded
  in `environment/curobo_source.json` and `environment/curobo-local.patch`.
  The original `LICENSE` is included. Unused sample robot meshes are omitted.
- `dual_v2_2_*`: project robot models, communication helpers and RT sources.
- `deployment/155_root`: exact calibrated robot configuration, installed
  camera service source, patched drivers and vendor CAN libraries required by
  the existing 155 installation.
- `deployment/camera_sdk`: installed Orbbec Python 3.10 Linux x86-64 SDK.
- `deployment/155_root/mnt/data/erazer/left_suction/realsense_sdk`: installed
  RealSense Python SDK for the D405 service.
- `deployment/155_root/usr/local/lib`: the two locally installed shared
  libraries linked by the archived RT (Pinocchio 2.6.10, Boost filesystem 1.83).

Keep this snapshot private unless the rights to all included project/vendor
components have been established. Python, CUDA, PyTorch, OS packages and their
licenses remain separate environment dependencies; full virtualenvs are not
included.
