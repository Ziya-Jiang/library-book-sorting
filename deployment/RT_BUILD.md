# RT source and binary provenance

Normal deployment restores `seldon_rt_control.xz`; its uncompressed SHA256 is
checked before installation and is part of the 63-file release contract.
This preserves the exact binary that matches both machines' current model and
protocol, without requiring historical build caches to start the system.

The corresponding current entry source is
`robot_cockpit/dual_arm_j6_continuous/rt/rt_control.cpp` (100 source tree), also
included byte-for-byte in the frozen 155 release payload. The archive contains:

- the legacy/native RT source tree under
  `155_root/home/erazer/cuarm_panel_control/cuarm_rt_control`;
- current tool and initialization headers under the profile `rt/`, plus
  `155_root/home/erazer/Seldon/dual_arm_cockpit/rt`;
- the installed velocity-feedback arm patch under
  `.../runtime/right_velocity_candidate_20261102/source`;
- the CAN feedback patch under
  `155_root/mnt/data/erazer/rt_can_feedback_20261104/candidate/source`;
- model/calibration files, vendor CAN libraries, Pinocchio/Boost runtime
  libraries, and exact build provenance in `rt_build_manifest.json`.

The historical profile `rt/build.py` relinks retained object files at absolute
paths on155. Those historical object caches are intentionally not exported;
that script is provenance, not a clean-machine build command. OS development
headers, Pinocchio/Boost/YAML/Eigen development installations and toolchain
versions remain environment dependencies. A fresh source rebuild must use the
listed driver patches, generate a new binary and regenerate matching contracts
before activation. Such a rebuilt binary has not been validated by this export.

The two requested startup/task commands do **not** invoke `rt/build.py`. They
use the checksum-verified archived binary, so a clean deployment of the supplied
runtime files does not rely on the old build cache.
