from isaaclab.utils import configclass

from accrobotics_go2.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class Go2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to unitree go2
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        # terrain-specific configurations for Go2
        if self.scene.terrain.terrain_generator is not None:
            if hasattr(self.scene.terrain.terrain_generator, 'sub_terrains'):
                if "boxes" in self.scene.terrain.terrain_generator.sub_terrains:
                    self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
                if "random_rough" in self.scene.terrain.terrain_generator.sub_terrains:
                    self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
                    self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        
        # action scaling for Go2
        self.actions.joint_pos.scale = 0.25
        
        # update mass randomization for Go2
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        
        # adjust rewards for Go2
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        if hasattr(self.rewards, 'dof_torques_l2'):
            self.rewards.dof_torques_l2.weight = -0.0002
        if hasattr(self.rewards, 'dof_acc_l2'):
            self.rewards.dof_acc_l2.weight = -2.5e-7


@configclass
class Go2RoughEnvCfg_PLAY(Go2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
