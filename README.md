# Library book sorting — continuous J6

当前连续 J6 图书回拨、抓取流程的独立部署快照，导出日期 **2026-11-15**。
保留当前运行算法和保护逻辑，覆盖下面两个入口的依赖：

```bash
bash /mnt/ssd/user/V/right_arm_j6_continuous_start.sh --restart-planner
bash /mnt/ssd/user/V/right_arm_j6_continuous.sh --execute
```

第一条准备相机、155 双臂 RT、工作台和100上的 CuRobo；第二条会接管并执行实际运动。
这是特定机器人、网络和标定的运行快照，不是通用机器人示例。

## 当前分工

- **100（10.30.3.100）**：任务编排、OpenAI API 请求、CuRobo GPU 规划、Rerun。
  当前 Planner 使用 GPU 0，NVIDIA RTX A6000 48 GB。
- **155（10.30.3.155，SSH 用户 erazer）**：左腕 D405 RGB-D、头部/右腕预览、
  双臂 RT、夹爪串口、实时反馈和网页后端。
- 当前任务的 Y、X、Z 和书本数量全部取自左腕 D405；头部只提供异步预览。
  GPT 使用代码里的 `gpt-6.1-sol`、`reasoning.effort=low`、`service_tier=priority`。
  模型权限及可用性由所用 API 账号决定。

## 包含内容

- `robot_cockpit/dual_arm_j6_continuous/`：连续控制、感知、准备求解、路径预检、
  同步反馈、Rerun、启动器、RT源码和当前参数。
- `dual_v2_2_agent/`、`robot_cockpit/launcher/`、`dual_v2_2_control/`：两个入口
  实际依赖的共享模块；保留导入依赖，不打包其他任务的运行数据。
- `dual_v2_2_configuration/`：协议/配置加载器和机器人网格；本版本独立 URDF、
  碰撞模型、关节限位位于当前 profile 的 `configuration/`。
- `deployment/`：155 相机服务、准确标定、已部署 backend release、驱动源码、
  必需原生库，以及 SHA256 锁定的压缩 RT 可执行文件。
- `third_party/curobo/`：现有 CuRobo 的源码与任务配置，含本机已有的一处补丁；
  不包含 CuRobo 示例机器人网格。版本、补丁及依赖见 `environment/`。
- `tools/`：只恢复文件的部署工具、离线检查；不会自动启动 RT 或发送目标。

不包含 API Key、SSH 密钥/密码、GUI 会话 token、历史 encoder 起点文件、
任务日志、实拍图、Rerun录制、模型API历史响应、训练权重或完整虚拟环境。
第三方版权说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 已有100/155环境上恢复

保留既有工作区。先在独立目录 clone 审阅，不要覆盖正在运行的 `/mnt/ssd/user/V`。
如果是原系统备份恢复，在正常结束控制、完成备份后，将本仓库部署到
**`/mnt/ssd/user/V`**。代码及模型契约包含该绝对路径；本次导出未擅自改写路径。

```bash
git clone https://github.com/Ziya-Jiang/library-book-sorting.git library-book-sorting
cd library-book-sorting
python3 tools/restore_runtime.py
/mnt/ssd/robotarm/env/robotram_wx/bin/python -B tools/offline_check.py --task-imports
```

`restore_runtime.py` 只补回被 `.gitignore` 排除的不可变 RT 文件、build manifest、
active-release 指针和空运行目录。不同内容的已有文件会被拒绝覆盖。

155 已有匹配 release 和相机服务时不必重复安装。新155的部署步骤见下节。
完成环境、155部署和源码路径准备后，两个运行命令仍是文首的原命令。

API Key 在100交互终端输入，或在该终端设置（示例不会把真实Key写入源码）：

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
bash /mnt/ssd/user/V/right_arm_j6_continuous_start.sh --restart-planner
bash /mnt/ssd/user/V/right_arm_j6_continuous.sh --execute
```

启动器的 `--check` 核验运行服务；`--stop-planner` 只正常停止本版本空闲Planner。
任务入口不带 `--execute` 仍会读取相机并调用API，**不是**纯离线测试。

## 新环境准备

目标为 Linux x86-64、Python 3.10。原生配置加载器和相机 SDK 是该 ABI。
100需要现有兼容 NVIDIA 驱动、CUDA 12.8 开发环境、g++、Eigen3、SSH、rsync；
155需要USB相机/CAN/串口访问权限和原硬件连接。预编译 RT 来自 glibc 2.35 环境。

### 100 Python环境

已有环境直接复用，不执行下面的安装操作。全新环境使用两个独立 venv：

```bash
python3.10 -m venv /mnt/ssd/robotarm/env/robotram_wx
/mnt/ssd/robotarm/env/robotram_wx/bin/python -m pip install -r environment/requirements-100-task.txt

python3.10 -m venv /mnt/ssd/robotarm/env/curobo
/mnt/ssd/robotarm/env/curobo/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
/mnt/ssd/robotarm/env/curobo/bin/python -m pip install -r environment/requirements-100-planner.txt ninja
SETUPTOOLS_SCM_PRETEND_VERSION=0.7.7.post1.dev5 /mnt/ssd/robotarm/env/curobo/bin/python -m pip install --no-build-isolation -e third_party/curobo
```

CuRobo安装需要 `nvcc`，不能仅安装运行时CUDA。准确环境记录见
`environment/100-planner-platform.json`。这些重建安装命令本次未在干净系统重新执行；
导出检查使用当前已安装环境。

### 155文件和环境

从100上传（仅传文件，不启停任何服务）：

```bash
bash tools/upload_155.sh erazer@10.30.3.155
ssh erazer@10.30.3.155
cd ~/library-book-sorting-package
python3 tools/install_155.py
```

先查看 plan。安装器拒绝覆盖任何不同内容的已有文件，避免覆盖活动配置。
全新155上先准备目录及系统动态库（管理员操作仅限系统库和目录）：

```bash
sudo apt-get install python3.10-venv liburdfdom-model3.0 libconsole-bridge1.0 libtinyxml2.6.2v5 libusb-1.0-0 libnl-route-3-200 rsync
sudo install -d -o erazer -g erazer /mnt/data/erazer
sudo python3 tools/install_155.py --system-libs-only --apply
sudo ldconfig
python3 tools/install_155.py --skip-system-libs --apply
python3.10 -m venv --system-site-packages /home/erazer/Seldon/keyboard_cockpit/.venv
/home/erazer/Seldon/keyboard_cockpit/.venv/bin/python -m pip install -r environment/requirements-155.txt
python3.10 -m venv --system-site-packages /home/erazer/deepseek_camera_tool/.venv
/home/erazer/deepseek_camera_tool/.venv/bin/python -m pip install -r environment/requirements-155.txt
systemctl --user daemon-reload
systemctl --user enable --now left-suction-d405.service
systemctl --user start deepseek-camera-tool.service
```

上述最后两条只启动相机服务；机械臂由100上的连续J6启动器准备。
左腕服务端口8793，预览共享源8792，Vision Tool8796，工作台8810；
100经SSH访问工作台8811，Planner任务10175/10176，反馈155:30240→100:30241。
需保证100到155的网卡源地址为10.30.3.100。

安装时恢复的 RT SHA256：
`36691a5776224fa9315400f9a5392ea1d868c4ba7de7a97a44c84eb6efadcdb9`。
RT压缩文件解压后约134MiB，使用压缩文件避免GitHub单文件大小限制。
保留运行中的源码/二进制/模型契约组合，不以重新编译的不同二进制冒充相同契约。
RT构建来源与当前重建边界见 [deployment/RT_BUILD.md](deployment/RT_BUILD.md)。

## 关键参数位置

- `continuous_trajectory.py`：当前连续执行速度系数 **0.8**；+X另有独立限速。
- `pre_sweep_lift.py`：本轮左腕计数 N，抬升 **(N+1)×10mm**。
- `left_wrist_spine.py`：左腕分割、RGB-D几何、抓取Z补偿 **−40mm**。
- `j6_only.py`、`j6_geometry.py`：J6下拨及单关节检查。
- `preparation_target.py`：候选顺序、实际Y→X→抬升预检与准备连接验证。
- `task_config/right_first_reset.json`：固定复位关节目标。
- 所有上述文件均在 `robot_cockpit/dual_arm_j6_continuous/`。

## 验证范围

导出时两个入口 `--help`、主流程/控制模块隔离导入、左腕标定hash、板端模型、
63项155发布契约、源码hash、Python语法检查和24项已有离线测试通过。
安装器已在 `/tmp` 的隔离根目录恢复并核验release；未部署到活动155目录。
本次没有调用付费API、启动/停止机器人、发布运动目标，**没有宣称完成干净机器的
全流程实机验证**。已有源码的算法行为和失败保护保持原样。

```bash
/mnt/ssd/robotarm/env/robotram_wx/bin/python -B tools/offline_check.py --task-imports
/mnt/ssd/robotarm/env/robotram_wx/bin/python -B -m unittest discover -s robot_cockpit/dual_arm_j6_continuous/tests -p 'test_*.py' -v
```
