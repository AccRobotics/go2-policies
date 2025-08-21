"""
Contains base tasks from official base Isaac Lab tasks for comparison purposes.
"""

import gymnasium as gym

import isaaclab_tasks.manager_based.locomotion.velocity.config.go2 as go2

gym.register(
    id="Base-Velocity-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{go2.__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{go2.agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Base-Velocity-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{go2.__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{go2.agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Base-Velocity-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{go2.__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{go2.agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Base-Velocity-Rough-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{go2.__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{go2.agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)