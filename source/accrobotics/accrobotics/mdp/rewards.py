from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def foot_deceleration_swing_phase(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.5,
    min_air_time: float = 0.05,
    deceleration_phase: float = 0.1,
    debug: bool = False,
    debug_print_freq: int = 100
) -> torch.Tensor:
    """Reward feet for decelerating only during the final phase of swing to preserve air time.
    
    This improved function only applies deceleration rewards when feet have been airborne 
    for a minimum duration, ensuring longer steps while still encouraging gentle landings.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.
        asset_cfg: Configuration for the robot asset to get body velocities.
        velocity_threshold: Maximum desired foot velocity during deceleration phase (m/s).
        min_air_time: Minimum air time before deceleration is encouraged (s). Typical values: 0.05-0.2.
        deceleration_phase: Duration of final swing phase where deceleration is rewarded (s).
        debug: If True, enables debug logging.
        debug_print_freq: Frequency of debug prints (every N steps).
    
    Returns:
        Reward tensor for foot deceleration behavior that preserves air time.
    """
    # Get contact sensor data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get robot articulation
    robot = env.scene[asset_cfg.name]
    
    # Get foot velocities in world frame
    foot_velocities = robot.data.body_lin_vel_w[:, sensor_cfg.body_ids, :]  # [num_envs, num_feet, 3]
    foot_speeds = torch.norm(foot_velocities, dim=-1)  # [num_envs, num_feet]
    
    # Get current air time for each foot
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    
    # Only consider feet that have been airborne for sufficient time
    sufficient_air_time = current_air_time > min_air_time
    
    # Identify feet in the final deceleration phase (approaching landing)
    # This is when air time is above min threshold but we're approaching expected landing
    in_deceleration_phase = (current_air_time > min_air_time) & (current_air_time <= min_air_time + deceleration_phase)
    
    # Get feet that just made contact after sufficient air time
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    good_landing = first_contact & (current_air_time > min_air_time)
    
    # Reward low velocities during deceleration phase and at good landings
    velocity_reward = torch.exp(-foot_speeds / velocity_threshold)
    
    # Apply reward only during appropriate phases
    phase_reward = velocity_reward * (in_deceleration_phase.float() * 0.3 + good_landing.float() * 0.7)
    
    # Sum reward across all feet
    reward = torch.sum(phase_reward, dim=1)
    
    # Debug logging
    if debug:
        if hasattr(env, '_debug_step_count'):
            env._debug_step_count += 1
        else:
            env._debug_step_count = 1
            
        if env._debug_step_count % debug_print_freq == 0:
            env_idx = 0  # Debug first environment
            print(f"\n=== Foot Deceleration Debug (Step {env._debug_step_count}) ===")
            print(f"Air times: {current_air_time[env_idx].cpu().numpy()}")
            print(f"Foot speeds: {foot_speeds[env_idx].cpu().numpy()}")
            print(f"In deceleration phase: {in_deceleration_phase[env_idx].cpu().numpy()}")
            print(f"Good landings: {good_landing[env_idx].cpu().numpy()}")
            print(f"Total reward (env 0): {reward[env_idx].item():.6f}")
            
            # Global statistics
            feet_with_sufficient_air = sufficient_air_time.sum().item()
            feet_in_decel_phase = in_deceleration_phase.sum().item()
            good_landings_count = good_landing.sum().item()
            
            print(f"Global: {feet_with_sufficient_air} sufficient air, {feet_in_decel_phase} in decel phase, {good_landings_count} good landings")
            print(f"Air time range: [{current_air_time.min().item():.3f}, {current_air_time.max().item():.3f}]")
    
    return reward
