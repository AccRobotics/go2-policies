
# Go2 Policies for Unitree Robot

This repository contains an Isaac Lab extension for training a variety of reinforcement learning policies for the Unitree Go2 robot.

## Policies and Tasks

| Policy Name | Isaac Lab Task | Description |
|:-:|:-:|:-:|
|  |  |  |

## Running Policies

The recommended way to run and train policies is via Docker, using the official Nvidia Isaac Lab base image.

### Prerequisites

Before running the Docker container, you need to authenticate with the Nvidia Container Registry to pull the Isaac Lab base image:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your-ngc-api-key>
```

You can get an NGC API key from the [Nvidia NGC portal](https://ngc.nvidia.com/setup/api-key).

### Quick Start

1. **Clone this repository**:

    ```bash
    git clone https://github.com/AccRobotics/go2-policies.git
    cd go2-policies
    ```

2. **Build and run the container**:

    ```bash
    cd docker
    # Edit .env.base as needed, then:
    docker compose --env-file .env.base --file docker-compose.yaml up --build
    ```

3. **Train or run a policy** inside the container (for instance via `docker exec -it accrobotics-isaaclab bash`):

    ```bash
    # Example: train a policy (replace with your actual command)
    python scripts/rsl_rl/train.py --task=<Your-Task-Name>
    ```

4. **Monitor training with TensorBoard**:

    TensorBoard is automatically started as part of the Docker Compose setup and is accessible at [http://localhost:6006](http://localhost:6006).

## Sim2Real Transfer

Instructions for sim2real transfer are to be determined (TBD).

