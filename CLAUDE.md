# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an Isaac Lab extension for training reinforcement learning policies for the Unitree Go2 robot. See README.md for general project information and setup instructions.

## Project Structure (for Claude's reference)

- `source/accrobotics_go2/accrobotics_go2/` - Main Python package
  - `tasks/go2/base.py` - Baseline environments from official Isaac Lab 
  - `tasks/go2/quiet.py` - Custom quiet locomotion environments with foot deceleration rewards
  - `mdp/rewards.py` - Custom reward functions including `foot_deceleration_swing_phase`
- `scripts/rsl_rl/` - Training scripts (train.py, play.py)

## Commands for Claude

### Python Execution
All Python commands must use Isaac Lab's wrapper. Inside the container, use the `python` alias:
```bash
# Training
python scripts/rsl_rl/train.py --task=<task-name> --headless
python scripts/rsl_rl/train.py --task=<task-name> --num_envs 4096 --max_iterations 1000

# Inference/Testing  
python scripts/rsl_rl/play.py --task=<task-name> --load_run=<run_name>

# Code quality checks
isort source/
pyright source/
```

### Task Names (see README.md for full list)
- Base tasks: `Base-Velocity-{Flat|Rough}-Unitree-Go2-v0`
- Quiet locomotion: `Acc-QuietVelocity-{Flat|Rough}-Unitree-Go2-v0` 
- Add `-Play-v0` suffix for inference/evaluation versions

## Implementation Notes

### Custom Rewards
The `foot_deceleration_swing_phase` reward in `mdp/rewards.py`:
- Encourages foot deceleration before landing to reduce impact noise
- Uses parameters: `velocity_threshold` (0.3), `min_air_time` (0.05), `deceleration_phase` (0.1)
- Applies reward during final swing phase and at good landings
- Includes debug logging capabilities

### Environment Structure
- Base environments (`base.py`) register official Isaac Lab configurations for comparison
- Quiet environments (`quiet.py`) extend base configs by adding the foot deceleration reward
- All environments follow Isaac Lab's `@configclass` pattern with `__post_init__` methods