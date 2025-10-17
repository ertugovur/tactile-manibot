# Tactile-Manibot: Tactile suite for Manibot

[ ![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Environment Setup](https://github.com/robot-dexterity/tactile-gym-3/actions/workflows/setup_env.yml/badge.svg)
![Scripts](https://github.com/robot-dexterity/tactile-gym-3/actions/workflows/script-tests.yml/badge.svg)

## Installation

This repo has only been developed and tested with Ubuntu 22.04 and python 3.10.
We use `uv` to manage the python environment.

Clone the repository:

```console
git clone https://github.com/robot-dexterity/tactile-manibot
cd tactile-manibot
```

Check if you have `uv` installed:

```sh
which uv
```

if you don't see output like: `/home/user/.local/bin/uv`, then [install ](https://docs.astral.sh/uv/getting-started/installation/)`uv`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set the environment up with:

```sh
uv sync
```

Run from within VScode or from command line:

```sh
uv run python tg3/utils/demo_predict.py -s sensor_name -source camera_source 
```

sensor names: 
robotiq_1,robotiq_2,robotiq_end_1,robotiq_end_2,airport_1,airport_2

camera_source default is 0, check which channel (X) is used on /dev/videoX. 
For example, run ffplay /dev/video0 and confirm that is the correct tactile image. 

