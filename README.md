<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/gym-ksuite/main/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Update Stompy S3 Model](https://github.com/kscalelabs/sim/actions/workflows/update_stompy_s3.yml/badge.svg)](https://github.com/kscalelabs/sim/actions/workflows/update_stompy_s3.yml)

</div>
<h1 align="center">
    <p>K-Scale Manipulation Suite</p>
</h1>

## Gymnasium+MuJoCo Environments

<table>
  <tr>
    <td><img src="assets/solo_arm.png" width="100%" alt="KManipSoloArm Env"/></td>
    <td><img src="assets/dual_arm.png" width="100%" alt="KManipDualArm Env"/></td>
    <td><img src="assets/full_body.png" width="100%" alt="KManipTorso Env"/></td>
  </tr>
  <tr>
    <td align="center"><b>KManipSoloArm</b> environment has one 7dof arm with a 1dof gripper. <b>KManipSoloArmVision</b> has a gripper cam, a head cam, and an overhead cam.</td>
    <td align="center"><b>KManipDualArm</b> environment has two 7dof arms with 1dof grippers. <b>KManipDualArmVision</b> has 2 gripper cams, a head cam, and an overhead cam.</td>
    <td align="center"><b>KManipTorso</b> environment has a 2dof head, two 6dof arms with 1dof grippers. <b>KManipTorsoVision</b> has 2 gripper cams, a head cam, and an overhead cam.</td>
  </tr>
</table>


## Setup

clone and install dependencies

```bash
git clone https://github.com/kscalelabs/gym-kmanip.git && cd gym-kmanip
conda create -y -n gym-kmanip python=3.10 && conda activate gym-kmanip
pip install -e .
```

run tests to verify installation

```bash
pip install pytest
pytest
```

for installation on the edge, just install on bare metal

```bash
sudo apt-get install libhdf5-dev
git clone https://github.com/kscalelabs/gym-kmanip.git && cd gym-kmanip
pip install -e .
```

## Getting Started

visualize one of the mujoco environments using the viewer

```bash
python gym_kmanip/examples/1_view_env.py
```

alternatively mujoco provides a nice standalone visualizer

[download standalone mujoco](https://github.com/google-deepmind/mujoco/releases)

```
tar -xzf ~/Downloads/mujoco-3.1.5-linux-x86_64.tar.gz -C /path/to/mujoco-3.1.5
/path/to/mujoco-3.1.5/bin/simulate gym_kmanip/assets/_env_solo_arm.xml
```

## Recording Data

data can be recorded to a rerun `.rrd` file for visualization

```bash
python gym_kmanip/examples/record_to_rrd.py
rerun gym_kmanip/data/foo/episode_1.rrd
```

data can be recorded to a h5py `.hdf5` file for training models

```bash
python gym_kmanip/examples/record_to_hdf5.py
```

you can also record data to a video file for sharing on social media

```bash
python gym_kmanip/examples/record_to_mp4.py
```

## Teleop

teleop can be used to control the robot and optionally record datasets

🤗 [K-Scale HuggingFace Datasets](https://huggingface.co/kscalelabs)

you will need additional dependencies

```bash
pip install vuer==0.0.30
```

start the server on the robot computer

```bash
python gym_kmanip/examples/4_record_data_teleop.py
```

start ngrok on the robot computer.

```bash
ngrok http 8012
```

open the browser app on the vr headset and go to the ngrok url

## Real Robot

the real robot works just like any other environment

you will need additional dependencies

```bash
pip install opencv-python==4.9.0.80
```

## Help Wanted

✅ solo arm w/ vision

✅ dual arm w/ vision

✅ torso w/ vision

✅ inverse kinematics using mujoco

⬜️ tune and improve ik

⬜️ recording dataset via teleop

⬜️ training policy from dataset

⬜️ evaluating policy on robot

## Dependencies

- [Gymnasium](https://gymnasium.farama.org/) is used for environment
- [MuJoCo](http://www.mujoco.org/) is used for physics simulation
- [PyTorch](https://pytorch.org/) is used for model training
- [Rerun](https://github.com/rerun-io/rerun/) is used for visualization
- [H5Py](https://docs.h5py.org/en/stable/) is used for logging datasets
- [HuggingFace](https://huggingface.co/) is used for dataset & model storage 
- [Vuer](https://github.com/vuer-ai/vuer) *teleop only* is used for visualization
- [ngrok](https://ngrok.com/download) *teleop only* is used for networking

helpful links and repos

- [dm_control](https://github.com/google-deepmind/dm_control)
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- [gym-aloha](https://github.com/huggingface/gym-aloha)
- [lerobot](https://github.com/huggingface/lerobot)
- [universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface)
- [loco-mujoco](https://github.com/robfiras/loco-mujoco)

### Citation

```
@misc{teleop-2024,
  title={gym-kmanip},
  author={Hugo Ponte},
  year={2024},
  url={https://github.com/kscalelabs/gym-kmanip}
}
```