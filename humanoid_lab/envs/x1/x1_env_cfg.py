# Copyright (c) 2024, AgiBot Inc. All rights reserved.
"""Isaac Sim / IsaacLab configuration for the X1 stand/walk task.

This is a port of `humanoid/envs/x1/x1_dh_stand_config.py` (Isaac Gym) to
IsaacLab 2.2 Direct-workflow configuration classes. All reward scales,
domain-randomization ranges, PD gains and observation parameters are kept
identical to the original pipeline.

Notes on deliberate differences (recorded as engineering assumptions):
1. Terrain: the original trimesh grid (flat/rough/slope/discrete mix) is
   reproduced with IsaacLab's height-field terrain generator using similar
   proportions and ranges.
2. Physics friction randomization of the robot shape material is skipped in
   phase 1 (URDF has no <dynamics> tags, and the original Isaac Gym code
   multiplied a zero default, i.e. effectively disabled joint friction/damping
   inside the solver; armature randomization is kept because it was assigned
   directly). The privileged-obs friction value stays randomized as before.
3. Joint friction/damping inside the solver are kept at 0 to match the
   effective behavior of the original pipeline; motor friction is emulated
   through the explicit Coulomb/viscous terms in the torque computation.
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, RigidBodyMaterialCfg, SimulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.schemas import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.height_field import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
)
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from humanoid_lab import LEGGED_GYM_ROOT_DIR


# ------------------------------------------------------------------ #
# Robot articulation: URDF -> USD at spawn time (cached per process) #
# ------------------------------------------------------------------ #
X1_URDF_PATH = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/urdf/x1.urdf"

X1_DEFAULT_JOINT_ANGLES = {
    'left_hip_pitch_joint': 0.4,
    'left_hip_roll_joint': 0.05,
    'left_hip_yaw_joint': -0.31,
    'left_knee_pitch_joint': 0.49,
    'left_ankle_pitch_joint': -0.21,
    'left_ankle_roll_joint': 0.0,
    'right_hip_pitch_joint': -0.4,
    'right_hip_roll_joint': -0.05,
    'right_hip_yaw_joint': 0.31,
    'right_knee_pitch_joint': 0.49,
    'right_ankle_pitch_joint': -0.21,
    'right_ankle_roll_joint': 0.0,
}

X1_ARTICULATION_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=UrdfFileCfg(
        asset_path=X1_URDF_PATH,
        activate_contact_sensors=True,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=True,
        replace_cylinders_with_capsules=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            drive_type="force",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=RigidBodyPropertiesCfg(),
        collision_props=CollisionPropertiesCfg(
            contact_offset=0.01,
            rest_offset=0.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos=X1_DEFAULT_JOINT_ANGLES,
    ),
    actuators={
        # explicit PD control computed in the env -> pure effort interface
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


@configclass
class X1SceneCfg(InteractiveSceneCfg):
    """Scene: procedural terrain + X1 articulation + contact sensor."""

    num_envs: int = 4096
    env_spacing: float = 8.0

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            curriculum=False,
            size=(8.0, 8.0),
            border_width=8.0,
            num_rows=20,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=1.5,
            use_cache=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                # flat patch (full platform) -- replaces original "flat" (30%)
                "flat": HfPyramidSlopedTerrainCfg(
                    proportion=0.3,
                    difficulty_range=(0.0, 0.0),
                    slope_range=(0.0, 0.0),
                    platform_width=8.0,
                ),
                # rough ground -- original rough flat (20%) + discrete (10%)
                "rough": HfRandomUniformTerrainCfg(
                    proportion=0.3,
                    noise_range=(0.005, 0.02),
                    noise_step=0.005,
                    difficulty_range=(0.0, 1.0),
                ),
                # slopes -- original slope up (20%) / slope down (20%)
                "slope_up": HfPyramidSlopedTerrainCfg(
                    proportion=0.2,
                    difficulty_range=(0.0, 1.0),
                    slope_range=(0.0, 0.1),
                    platform_width=3.0,
                ),
                "slope_down": HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.2,
                    difficulty_range=(0.0, 1.0),
                    slope_range=(0.0, 0.1),
                    platform_width=3.0,
                ),
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.6,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = X1_ARTICULATION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        history_length=0,
        track_air_time=False,
    )


# --------------------------------------------------------------------------- #
# Legacy-style plain config classes kept from the original pipeline           #
# (rewards / domain-rand / commands / noise / obs scales)                     #
# --------------------------------------------------------------------------- #
class CfgEnv:
    frame_stack = 66            # all history obs num
    short_frame_stack = 5       # short history steps
    c_frame_stack = 3           # all history privileged obs num
    num_single_obs = 47
    num_observations = frame_stack * num_single_obs
    single_num_privileged_obs = 73
    single_linvel_index = 53
    num_privileged_obs = c_frame_stack * single_num_privileged_obs
    num_actions = 12
    num_envs = 4096
    episode_length_s = 24.0
    init_state_z = 0.7          # base spawn height above env origin
    use_ref_actions = False
    num_commands = 5            # sin_pos cos_pos vx vy vz


class CfgSafety:
    pos_limit = 1.0
    vel_limit = 1.0
    torque_limit = 0.85


class CfgNoise:
    add_noise = True
    noise_level = 1.5

    class noise_scales:
        dof_pos = 0.02
        dof_vel = 1.5
        ang_vel = 0.2
        lin_vel = 0.1
        quat = 0.1
        gravity = 0.05
        height_measurements = 0.1


class CfgControl:
    control_type = 'P'
    stiffness = {'hip_pitch_joint': 30, 'hip_roll_joint': 40, 'hip_yaw_joint': 35,
                 'knee_pitch_joint': 100, 'ankle_pitch_joint': 35, 'ankle_roll_joint': 35}
    damping = {'hip_pitch_joint': 3, 'hip_roll_joint': 3.0, 'hip_yaw_joint': 4,
               'knee_pitch_joint': 10, 'ankle_pitch_joint': 0.5, 'ankle_roll_joint': 0.5}
    action_scale = 0.5
    decimation = 10             # 100 Hz policy at 1 kHz physics


class CfgTerrain:
    mesh_type = 'trimesh'
    curriculum = False
    measure_heights = False
    static_friction = 0.6
    dynamic_friction = 0.6
    terrain_length = 8.0
    terrain_width = 8.0
    num_rows = 20
    num_cols = 20
    max_init_terrain_level = 5
    platform = 3.0
    num_height = 0


class CfgDomainRand:
    randomize_friction = True
    friction_range = [0.2, 1.3]
    restitution_range = [0.0, 0.4]

    push_robots = True
    push_interval_s = 4
    update_step = 2000 * 24
    push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    max_push_vel_xy = 0.2
    max_push_ang_vel = 0.2

    randomize_base_mass = True
    added_mass_range = [-3, 3]

    randomize_com = True
    com_displacement_range = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]

    randomize_gains = True
    stiffness_multiplier_range = [0.8, 1.2]
    damping_multiplier_range = [0.8, 1.2]

    randomize_torque = True
    torque_multiplier_range = [0.8, 1.2]

    randomize_link_mass = True
    added_link_mass_range = [0.9, 1.1]

    randomize_motor_offset = True
    motor_offset_range = [-0.035, 0.035]

    randomize_joint_friction = False       # solver-side joint friction (kept 0, see module docstring)
    randomize_joint_damping = False        # solver-side joint damping (kept 0, see module docstring)
    randomize_joint_armature = True
    joint_armature_range = [0.0001, 0.05]

    add_lag = False
    randomize_lag_timesteps = False
    randomize_lag_timesteps_perstep = False
    lag_timesteps_range = [5, 40]

    add_dof_lag = True
    randomize_dof_lag_timesteps = True
    randomize_dof_lag_timesteps_perstep = False
    dof_lag_timesteps_range = [0, 40]

    add_dof_pos_vel_lag = False
    randomize_dof_pos_lag_timesteps = False
    randomize_dof_pos_lag_timesteps_perstep = False
    dof_pos_lag_timesteps_range = [7, 25]
    randomize_dof_vel_lag_timesteps = False
    randomize_dof_vel_lag_timesteps_perstep = False
    dof_vel_lag_timesteps_range = [7, 25]

    add_imu_lag = False
    randomize_imu_lag_timesteps = True
    randomize_imu_lag_timesteps_perstep = False
    imu_lag_timesteps_range = [1, 10]

    randomize_coulomb_friction = True
    joint_coulomb_range = [0.1, 0.9]
    joint_viscous_range = [0.05, 0.1]


class CfgCommands:
    curriculum = True
    max_curriculum = 1.5
    num_commands = 4
    resampling_time = 25.0
    gait = ["walk_omnidirectional", "stand", "walk_omnidirectional"]
    gait_time_range = {"walk_sagittal": [2, 6],
                       "walk_lateral": [2, 6],
                       "rotate": [2, 3],
                       "stand": [2, 3],
                       "walk_omnidirectional": [4, 6]}
    heading_command = False
    stand_com_threshold = 0.05
    sw_switch = True

    class ranges:
        lin_vel_x = [-0.4, 1.2]
        lin_vel_y = [-0.4, 0.4]
        ang_vel_yaw = [-0.6, 0.6]
        heading = [-3.14, 3.14]


class CfgRewards:
    soft_dof_pos_limit = 0.98
    soft_dof_vel_limit = 0.9
    soft_torque_limit = 0.9
    base_height_target = 0.61
    foot_min_dist = 0.2
    foot_max_dist = 1.0

    final_swing_joint_delta_pos = [0.25, 0.05, -0.11, 0.35, -0.16, 0.0, -0.25, -0.05, 0.11, 0.35, -0.16, 0.0]
    target_feet_height = 0.03
    target_feet_height_max = 0.06
    feet_to_ankle_distance = 0.041
    cycle_time = 0.7
    only_positive_rewards = True
    tracking_sigma = 5
    max_contact_force = 700

    class scales:
        ref_joint_pos = 2.2
        feet_clearance = 1.
        feet_contact_number = 2.0
        feet_air_time = 1.2
        foot_slip = -0.1
        feet_distance = 0.2
        knee_distance = 0.2
        feet_contact_forces = -0.01
        tracking_lin_vel = 1.8
        tracking_ang_vel = 1.1
        vel_mismatch_exp = 0.5
        low_speed = 0.2
        track_vel_hard = 0.5
        default_joint_pos = 1.0
        orientation = 1.
        feet_rotation = 0.3
        base_height = 0.2
        base_acc = 0.2
        action_smoothness = -0.002
        torques = -8e-9
        dof_vel = -2e-8
        dof_acc = -1e-7
        collision = -1.
        stand_still = 2.5
        dof_vel_limits = -1
        dof_pos_limits = -10.
        dof_torque_limits = -0.1


class CfgNormalization:
    class obs_scales:
        lin_vel = 2.
        ang_vel = 1.
        dof_pos = 1.
        dof_vel = 0.05
        quat = 1.
        height_measurements = 5.0
    clip_observations = 100.
    clip_actions = 100.


# PPO runner configuration (kept identical to X1DHStandCfgPPO)
class X1DHStandCfgPPO:
    seed = 5
    runner_class_name = 'DHOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims = [256, 128, 64]
        kernel_size = [6, 4]
        filter_size = [32, 16]
        stride_size = [3, 2]
        lh_output_dim = 64
        in_channels = CfgEnv.frame_stack

    class algorithm:
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
        lin_vel_idx = CfgEnv.single_num_privileged_obs * (CfgEnv.c_frame_stack - 1) + CfgEnv.single_linvel_index

    class runner:
        policy_class_name = 'ActorCriticDH'
        algorithm_class_name = 'DHPPO'
        num_steps_per_env = 24
        max_iterations = 20000
        save_interval = 100
        experiment_name = 'x1_dh_stand'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None


@configclass
class X1DHStandEnvCfg(DirectRLEnvCfg):
    """Direct-workflow env config: sim/scene wiring + legacy parameter groups."""

    # DirectRLEnvCfg fields
    decimation: int = CfgControl.decimation
    episode_length_s: float = CfgEnv.episode_length_s
    observation_space: int = CfgEnv.num_observations          # 66 * 47
    state_space: int = CfgEnv.num_privileged_obs              # 3 * 73 (critic)
    action_space: int = CfgEnv.num_actions                    # 12
    sim: SimulationCfg = SimulationCfg(
        dt=0.001,
        render_interval=10,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            gpu_max_rigid_contact_count=2**23,
        ),
    )
    scene: X1SceneCfg = X1SceneCfg()

    # legacy parameter groups (referenced by the env implementation)
    env = CfgEnv()
    safety = CfgSafety()
    noise = CfgNoise()
    control = CfgControl()
    terrain = CfgTerrain()
    domain_rand = CfgDomainRand()
    commands = CfgCommands()
    rewards = CfgRewards()
    normalization = CfgNormalization()
    obs_scales = CfgNormalization.obs_scales()
