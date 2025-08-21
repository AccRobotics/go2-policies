"""
Quiet locomotion task environments for the Unitree Go2 robot.

This module contains environment configurations that teach the Go2 robot to walk
more quietly by incentivizing foot deceleration before landing while maintaining
minimum air time for natural gait patterns.

The environments extend the base Go2 configurations with additional reward terms
focused on gentle foot landings to reduce impact noise during locomotion.

Available environments:
- Acc-QuietVelocity-Flat-Unitree-Go2-v0: Training on flat terrain
- Acc-QuietVelocity-Flat-Unitree-Go2-Play-v0: Evaluation on flat terrain  
- Acc-QuietVelocity-Rough-Unitree-Go2-v0: Training on rough terrain
- Acc-QuietVelocity-Rough-Unitree-Go2-Play-v0: Evaluation on rough terrain
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
    UnitreeGo2FlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
    UnitreeGo2RoughEnvCfg_PLAY,
)

import accrobotics.mdp as mdp

# Shared reward term for foot deceleration to keep code simple and consistent
foot_deceleration_swing_phase = RewTerm(
    func=mdp.foot_deceleration_swing_phase,
    weight=0.25,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        "asset_cfg": SceneEntityCfg("robot"),
        "velocity_threshold": 0.3,
        "min_air_time": 0.05,  # Ensure minimum air time before dec
        "deceleration_phase": 0.1,  # Duration of final swing phase for deceleration
    }
)

# Below are the environment modifications for the Go2 robot to learn quieter walking

@configclass
class QuietRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.foot_deceleration = foot_deceleration_swing_phase

@configclass
class QuietRoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.foot_deceleration = foot_deceleration_swing_phase

@configclass
class QuietFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.foot_deceleration = foot_deceleration_swing_phase

@configclass
class QuietFlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.foot_deceleration = foot_deceleration_swing_phase

# Below is boilerplate code to register the environments with Gym
import gymnasium as gym

import isaaclab_tasks.manager_based.locomotion.velocity.config.go2 as go2

gym.register(
    id="Acc-QuietVelocity-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:QuietFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{go2.agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Acc-QuietVelocity-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:QuietFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{go2.agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Acc-QuietVelocity-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:QuietRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{go2.agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Acc-QuietVelocity-Rough-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:QuietRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{go2.agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)