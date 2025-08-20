# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an Isaac Lab extension for training reinforcement learning policies for the Unitree Go2 robot. The project uses RSL-RL (Robotics Systems Lab - Reinforcement Learning) for training locomotion policies with Isaac Sim as the physics simulator.

## Project Structure

- `source/accrobotics_go2/` - Main Python package containing the Go2 RL tasks
  - `tasks/locomotion/velocity/` - Velocity-based locomotion task implementations
  - `config/go2/` - Go2-specific configuration files for flat and rough terrain
  - `mdp/` - Markov Decision Process components (rewards, curriculums)
- `scripts/rsl_rl/` - Training and inference scripts
  - `train.py` - Main training script using RSL-RL
  - `play.py` - Inference/evaluation script with policy export
- `docker/` - Docker containerization files
- `logs/` - Training logs and model checkpoints (organized by task and timestamp)

## Common Commands

### Development Environment
The recommended development approach is via Docker using the official Nvidia Isaac Lab base image:

```bash
# Build and run the development container
cd docker
docker compose --env-file .env.base --file docker-compose.yaml up --build

# Enter the running container
docker exec -it accrobotics-isaaclab bash
```

### Python Usage
Python is available through Isaac Lab's wrapper script. Commands can be run either inside the container (using the `python` alias) or from outside:

```bash
# From outside the container
docker exec accrobotics-isaaclab /workspace/isaaclab/_isaac_sim/python.sh --version

# Inside the container (interactive session)
docker exec -it accrobotics-isaaclab bash
python --version  # Uses alias from bashrc
```

### Training Policies
```bash
# From outside container
docker exec accrobotics-isaaclab /workspace/isaaclab/_isaac_sim/python.sh scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-Go2-v0 --headless

# Inside container (using alias)
python scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-Go2-v0 --headless
python scripts/rsl_rl/train.py --task=Isaac-Velocity-Rough-Go2-v0 --headless
python scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-Go2-v0 --video
python scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-Go2-v0 --num_envs 4096 --max_iterations 1000 --headless
```

### Running/Testing Trained Policies
```bash
# From outside container
docker exec accrobotics-isaaclab /workspace/isaaclab/_isaac_sim/python.sh scripts/rsl_rl/play.py --task=Isaac-Velocity-Flat-Go2-v0 --load_run=<run_name> --livestream=2

# Inside container (using alias)
python scripts/rsl_rl/play.py --task=Isaac-Velocity-Flat-Go2-v0 --load_run=<run_name> --livestream=2
python scripts/rsl_rl/play.py --task=Isaac-Velocity-Flat-Go2-v0 --load_run=<run_name> --video
```

### Code Quality
```bash
# From outside container
docker exec accrobotics-isaaclab /workspace/isaaclab/_isaac_sim/python.sh -m isort source/
docker exec accrobotics-isaaclab /workspace/isaaclab/_isaac_sim/python.sh -m pyright source/

# Inside container (using aliases)
isort source/  # Uses pip alias from bashrc
pyright source/
```

## Key Components

### Task Configuration
- Task configurations follow Isaac Lab patterns with environment, scene, and agent configs
- Go2 tasks are defined in `source/accrobotics_go2/accrobotics_go2/tasks/locomotion/velocity/`
- Flat vs rough terrain variants are configured in `config/go2/flat_env_cfg.py` and `config/go2/rough_env_cfg.py`

### Training Infrastructure
- Uses Hydra for configuration management
- RSL-RL integration through `RslRlVecEnvWrapper` and `OnPolicyRunner`
- Automatic model export to ONNX and PyTorch JIT formats during inference
- TensorBoard logging available at http://localhost:6006 when using Docker setup

### Logging and Checkpoints
- Training logs saved to `logs/rsl_rl/<experiment_name>/<timestamp>/`
- Model checkpoints saved as `model_<iteration>.pt`
- Exported policies saved in `exported/` subdirectory as `policy.onnx` and `policy.pt`
- Configuration files dumped as YAML and pickle files in `params/` subdirectory

## Available Tasks

- `Isaac-Velocity-Flat-Go2-v0` - Go2 locomotion on flat terrain
- `Isaac-Velocity-Rough-Go2-v0` - Go2 locomotion on rough terrain

## Docker Configuration

The project uses Isaac Lab's official Docker image as base. Key environment variables and paths are configured in `docker/.env.base`. TensorBoard is automatically started as a sidecar container for training monitoring.