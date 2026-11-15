# cuarm_rt_control
Ubuntu: 22.04
computer: Jetson Orin(Arm64)

communication board: USB to CAN
motor: usb_to_can
brake: usb_to_can
force sensor: usb_to_can

Example start:
```bash
sudo env LD_LIBRARY_PATH=/home/cuarm/dual_v1_panel_control/cuarm_rt_control/eyou_dual_arm/lib/eyou_harmonic/x86:LD_LIBRARY_PATH ./eyou_dual_v1_rt_control
```