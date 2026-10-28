# Third-party components

Nothing from these projects is vendored into this repository. The Menagerie model is
fetched at build time by `scripts/fetch_menagerie.sh`; LeRobot and the other Python
packages are pip dependencies pinned in `requirements.lock`.

| Component | Source | License | Changes |
|---|---|---|---|
| mujoco_menagerie `franka_emika_panda` (MJCF + meshes) | <https://github.com/google-deepmind/mujoco_menagerie> at `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` | Apache-2.0 | Not modified on disk. At load time (`MjSpec`) the env adds a TCP site and wrist camera to `hand`, sets `gravcomp=1` on the arm bodies, sets elliptic cones / impratio 10, disables shadows, and adds the table, cube, target and front camera. |
| LeRobot | <https://github.com/huggingface/lerobot> (PyPI `lerobot==0.6.1`) | Apache-2.0 | Used as a library and CLI (`LeRobotDataset`, `lerobot-train`, `ACTPolicy`, processor pipelines). No code copied; the env, recorder, harness and report are written here. |
| gym_hil | <https://github.com/huggingface/gym-hil> (PyPI `gym-hil==0.1.14`) | Apache-2.0 | Not used in v0.1. Planned as the key-layout reference for keyboard teleop (pattern only, no code). |
| MuJoCo | <https://github.com/google-deepmind/mujoco> (PyPI `mujoco==3.17.0`) | Apache-2.0 | Library dependency. |
| PyTorch / torchvision (CPU builds) | <https://pytorch.org> | BSD-3-Clause | Library dependency; ACT's ResNet-18 backbone starts from torchvision's ImageNet weights (`ResNet18_Weights.IMAGENET1K_V1`). |
