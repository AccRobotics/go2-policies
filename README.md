
# Go2 Policies for Unitree Robot

This repository contains an Isaac Lab extension for training a variety of reinforcement learning policies for the Unitree Go2 robot.

## Policies and Tasks

| Policy Name | Isaac Lab Task | Description |
|:-:|:-:|:-:|
|  |  |  |

## Running Policies

The recommended way to run and train policies is via Docker, using the Isaac Lab base image. The base image is best managed using the environment scripts provided in [accrobotics/envs](https://github.com/AccRobotics/envs).

### Quick Start

1. **Clone this repository** (and [accrobotics/envs](https://github.com/AccRobotics/envs) if not already):

    ```bash
    git clone https://github.com/AccRobotics/go2-policies.git
    git clone https://github.com/AccRobotics/envs.git
    ```

2. **Set up the Isaac Lab Docker environment** using the scripts in `accrobotics/envs` (see that repo for details).

3. **Build and run the container** for this project:

    ```bash
    cd go2-policies/docker
    # Edit .env.base as needed, then:
    docker compose --env-file .env.base --file docker-compose.yaml up --build
    ```

4. **Train or run a policy** inside the container:

    ```bash
    # Example: train a policy (replace with your actual command)
    python scripts/rsl_rl/train.py --task=<Your-Task-Name>
    ```

> **Note:** For best results, always use the Isaac Lab base image and environment setup from accrobotics/envs to ensure compatibility.

## Sim2Real Transfer

Instructions for sim2real transfer are to be determined (TBD).

