# Copyright (c) 2024, AgiBot Inc. All rights reserved.
"""X1 humanoid stand/walk environment on Isaac Sim / IsaacLab (direct workflow).

Faithful port of `humanoid/envs/x1/x1_dh_stand_env.py` + the parts of
`humanoid/envs/base/legged_robot.py` it relies on. Observation layout,
reward terms, gait scheduling, teacher-student history buffers, domain
randomization and the explicit PD torque controller are kept identical.

Isaac Gym -> IsaacLab mapping used here:
  gym.create_sim                       -> SimulationContext (handled by DirectRLEnv)
  ground plane / trimesh               -> TerrainImporter (generator terrain)
  create_env + URDF actor              -> Articulation with UrdfFileCfg spawn
  acquire/refresh *_state_tensor       -> ArticulationData / ContactSensorData
  set_dof_actuation_force_tensor       -> Articulation.set_joint_effort_target
  set_dof_state_tensor_indexed         -> Articulation.write_joint_state_to_sim
  set_actor_root_state_tensor[_idx]    -> Articulation.write_root_state_to_sim
  set_actor_dof_properties             -> write_joint_{armature,friction,damping}...
  set_actor_rigid_body_properties      -> root_physx_view.set_masses
"""

from __future__ import annotations

import numpy as np
import torch
from collections import deque

from isaaclab.envs import DirectRLEnv

from humanoid_lab.utils.torch_utils import (
    torch_rand_float,
    quat_rotate,
    quat_rotate_inverse,
    quat_apply,
    get_euler_xyz_tensor,
    get_axis_params,
    wrap_to_pi,
)

from .x1_env_cfg import X1DHStandEnvCfg


class X1DHStandEnv(DirectRLEnv):
    """X1 stand/walk environment (teacher-student observations)."""

    cfg: X1DHStandEnvCfg

    def __init__(self, cfg: X1DHStandEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # all IsaacLab buffers (scene entities, physics views, sensor data)
        # are ready after the base class finishes; allocate our buffers here
        self._init_buffers()

    # ------------------------------------------------------------------ #
    # Scene / buffer setup                                               #
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        # scene entities come from cfg.scene (terrain, robot, contact sensor)
        super()._setup_scene()

    def _init_buffers(self):
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_forces"]
        self._terrain = self.scene.terrain

        cfg = self.cfg
        # NOTE: num_envs / device are read-only properties in DirectRLEnv
        #       (they resolve to self.scene.num_envs / self.sim.device).
        self.num_dof = self._robot.num_joints
        self.num_bodies = self._robot.num_bodies
        self.num_actions = cfg.env.num_actions

        # ---------------- body indices (name based, order independent) ----
        self.feet_indices, _ = self._robot.find_bodies(".*ankle_roll")
        self.knee_indices, _ = self._robot.find_bodies(".*knee_pitch")
        self.penalised_contact_indices, _ = self._robot.find_bodies("base_link")
        # sanity: contact sensor must see the same body set as the articulation
        sensor_bodies = self._contact_sensor.num_instances if hasattr(self._contact_sensor, "num_instances") else 0
        if sensor_bodies != self._robot.num_bodies:
            print(
                f"[X1DHStandEnv] WARNING contact sensor bodies ({sensor_bodies}) "
                f"!= articulation bodies ({self._robot.num_bodies}); "
                "contact indexing may be misaligned"
            )

        # ---------------- joint ordering helpers --------------------------
        self.dof_names = list(self._robot.joint_names)
        default_angles = self._robot.data.default_joint_pos[0].clone()
        self.default_dof_pos = default_angles.unsqueeze(0)  # (1, num_dof)
        # PD gains from cfg by name substring (same logic as original)
        p_gains = torch.zeros(self.num_dof, device=self.device)
        d_gains = torch.zeros(self.num_dof, device=self.device)
        for i, name in enumerate(self.dof_names):
            found = False
            for dof_name, kp in cfg.control.stiffness.items():
                if dof_name in name:
                    p_gains[i] = kp
                    d_gains[i] = cfg.control.damping[dof_name]
                    found = True
            if not found and cfg.control.control_type in ["P", "V"]:
                print(f"[X1DHStandEnv] PD gain of joint {name} not defined, setting to zero")
        self.p_gains = p_gains.unsqueeze(0)
        self.d_gains = d_gains.unsqueeze(0)
        self.default_joint_pd_target = self.default_dof_pos.clone()

        # ---------------- limits (from USD, scaled by safety factors) -----
        pos_limits = self._robot.data.soft_joint_pos_limits[0].clone()  # (num_dof, 2)
        vel_limits = self._robot.data.joint_vel_limits[0].clone()
        effort_limits = self._robot.data.joint_effort_limits[0].clone()
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, device=self.device)
        lo = pos_limits[:, 0] * cfg.safety.pos_limit
        hi = pos_limits[:, 1] * cfg.safety.pos_limit
        self.dof_pos_limits[:, 0] = lo
        self.dof_pos_limits[:, 1] = hi
        # soft limits (mirrors original _process_dof_props)
        m = 0.5 * (lo + hi)
        r = hi - lo
        soft = cfg.rewards.soft_dof_pos_limit
        self.dof_pos_limits[:, 0] = m - 0.5 * r * soft
        self.dof_pos_limits[:, 1] = m + 0.5 * r * soft
        self.dof_vel_limits = vel_limits * cfg.safety.vel_limit
        self.torque_limits = effort_limits * cfg.safety.torque_limit

        # ---------------- common buffers ----------------------------------
        self.gravity_vec = torch.tensor(get_axis_params(-1.0, 2), device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_torques = torch.zeros_like(self.torques)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.commands = torch.zeros(self.num_envs, cfg.commands.num_commands, device=self.device)
        self.commands_scale = torch.tensor(
            [cfg.obs_scales.lin_vel, cfg.obs_scales.lin_vel, cfg.obs_scales.ang_vel],
            device=self.device,
        )
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)

        self.command_ranges = {
            "lin_vel_x": [float(v) for v in cfg.commands.ranges.lin_vel_x],
            "lin_vel_y": [float(v) for v in cfg.commands.ranges.lin_vel_y],
            "ang_vel_yaw": [float(v) for v in cfg.commands.ranges.ang_vel_yaw],
            "heading": [float(v) for v in cfg.commands.ranges.heading],
        }

        self.rand_push_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), device=self.device)

        # domain randomization state buffers
        dr = cfg.domain_rand
        self.payload_masses = torch_rand_float(*dr.added_mass_range, (self.num_envs, 1), device=self.device) \
            if dr.randomize_base_mass else torch.zeros(self.num_envs, 1, device=self.device)
        self.link_masses = torch_rand_float(*dr.added_link_mass_range, (self.num_envs, self.num_bodies), device=self.device) \
            if dr.randomize_link_mass else torch.ones(self.num_envs, self.num_bodies, device=self.device)
        self.com_displacements = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_frictions = torch_rand_float(*dr.friction_range, (self.num_envs, 1), device=self.device) \
            if dr.randomize_friction else torch.ones(self.num_envs, 1, device=self.device)
        self.body_mass = torch.zeros(self.num_envs, device=self.device)
        self.torque_multi = torch.ones(self.num_envs, self.num_actions, device=self.device)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.randomized_p_gains = self.p_gains.repeat(self.num_envs, 1).clone()
        self.randomized_d_gains = self.d_gains.repeat(self.num_envs, 1).clone()
        self.randomized_joint_coulomb = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.randomized_joint_viscous = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.joint_armatures = torch_rand_float(*dr.joint_armature_range, (self.num_envs, self.num_actions), device=self.device) \
            if dr.randomize_joint_armature else torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.joint_friction_coeffs = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.joint_damping_coeffs = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.init_body_mass = self._robot.root_physx_view.get_masses()[0, 0].item()

        # gait scheduling buffers
        self.gait_time = torch.zeros(self.num_envs, len(cfg.commands.gait), dtype=torch.int, device=self.device)
        self.phase_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_start = torch.randint(0, 2, (self.num_envs,), device=self.device) * 0.5

        # teacher-student history buffers
        self.obs_history = deque(maxlen=cfg.env.frame_stack)
        self.critic_history = deque(maxlen=cfg.env.c_frame_stack)
        for _ in range(cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(self.num_envs, cfg.env.num_single_obs, device=self.device))
        for _ in range(cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(self.num_envs, cfg.env.single_num_privileged_obs, device=self.device)
            )

        # dof lag buffer (q + dq history for observation delay emulation)
        if dr.add_dof_lag:
            self.dof_lag_buffer = torch.zeros(
                self.num_envs, self.num_actions * 2, dr.dof_lag_timesteps_range[1] + 1, device=self.device
            )
            self.dof_lag_timestep = torch.randint(
                dr.dof_lag_timesteps_range[0], dr.dof_lag_timesteps_range[1] + 1,
                (self.num_envs,), device=self.device,
            )
            self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
        if dr.add_lag:
            self.lag_buffer = torch.zeros(
                self.num_envs, self.num_actions, dr.lag_timesteps_range[1] + 1, device=self.device
            )
            self.lag_timestep = torch.randint(
                dr.lag_timesteps_range[0], dr.lag_timesteps_range[1] + 1,
                (self.num_envs,), device=self.device,
            )

        # noise
        self.add_noise = cfg.noise.add_noise
        self.noise_scale_vec = self._get_noise_scale_vec(cfg)

        # last states
        self.last_rigid_state = torch.zeros(self.num_envs, self.num_bodies, 13, device=self.device)
        self.last_contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device)

        # feet reward helpers
        self.last_feet_z = cfg.rewards.feet_to_ankle_distance
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.ref_dof_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.ref_action = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.command_input = torch.zeros((self.num_envs, 5), device=self.device)

        # reward machinery
        self._prepare_reward_function()

        # one-shot rigid-body randomization (masses/inertias), mirrors env creation
        # randomization in the original Isaac Gym pipeline
        self.randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device))
        if self.cfg.domain_rand.randomize_joint_armature:
            self._robot.write_joint_armature_to_sim(self.joint_armatures)

    # ------------------------------------------------------------------ #
    # Derived state views (IsaacLab tensors, world frame)                #
    # IsaacLab exposes quaternions in wxyz; the original pipeline uses   #
    # xyzw everywhere, so we convert at the boundary.                    #
    # ------------------------------------------------------------------ #
    @property
    def root_states(self) -> torch.Tensor:
        """(num_envs, 13): pos(3) quat_xyzw(4) linvel(3) angvel(3) in world."""
        rs = self._robot.data.root_state_w.clone()
        rs[:, 3:7] = rs[:, 3:7][:, [1, 2, 3, 0]]  # wxyz -> xyzw
        return rs

    @property
    def base_quat(self) -> torch.Tensor:
        # root_state_w stores quaternion in xyzw convention (Isaac Gym parity)
        return self.root_states[:, 3:7]

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._robot.data.joint_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._robot.data.joint_vel

    @property
    def contact_forces(self) -> torch.Tensor:
        return self._contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)

    @property
    def rigid_state(self) -> torch.Tensor:
        """(num_envs, num_bodies, 13): pos, quat(xyzw), linvel, angvel (world)."""
        data = self._robot.data
        rs = torch.cat(
            (data.body_pos_w, data.body_quat_w.clone()[..., [1, 2, 3, 0]], data.body_lin_vel_w, data.body_ang_vel_w),
            dim=-1,
        )
        return rs

    # ------------------------------------------------------------------ #
    # Step flow                                                          #
    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor):
        clip = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip, clip).to(self.device)
        if self.cfg.env.use_ref_actions:
            self.actions = self.actions + self.ref_action

    def _apply_action(self):
        """Called every physics substep (1 kHz): update lag buffer, recompute PD torques."""
        dr = self.cfg.domain_rand
        if dr.add_dof_lag:
            q = self.dof_pos
            dq = self.dof_vel
            self.dof_lag_buffer[:, :, 1:] = self.dof_lag_buffer[:, :, : dr.dof_lag_timesteps_range[1]].clone()
            self.dof_lag_buffer[:, :, 0] = torch.cat((q, dq), 1).clone()
        if dr.add_lag:
            actions_scaled = self.actions * self.cfg.control.action_scale
            self.lag_buffer[:, :, 1:] = self.lag_buffer[:, :, : dr.lag_timesteps_range[1]].clone()
            self.lag_buffer[:, :, 0] = actions_scaled.clone()

        self.torques = self._compute_torques(self.actions).view(self.num_envs, self.num_actions)
        self._robot.set_joint_effort_target(self.torques)

    def _compute_torques(self, actions):
        cfg = self.cfg
        dr = cfg.domain_rand
        actions_scaled = actions * cfg.control.action_scale
        if dr.add_lag:
            self.lagged_actions_scaled = self.lag_buffer[
                torch.arange(self.num_envs, device=self.device), :, self.lag_timestep.long()
            ]
        else:
            self.lagged_actions_scaled = actions_scaled

        if dr.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if dr.randomize_coulomb_friction:
            torques = (
                p_gains * (self.lagged_actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets)
                - d_gains * self.dof_vel
                - self.randomized_joint_viscous * self.dof_vel
                - self.randomized_joint_coulomb * torch.sign(self.dof_vel)
            )
        else:
            torques = p_gains * (
                self.lagged_actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets
            ) - d_gains * self.dof_vel

        if dr.randomize_torque:
            self.torque_multi = torch_rand_float(
                dr.torque_multiplier_range[0], dr.torque_multiplier_range[1],
                (self.num_envs, self.num_actions), device=self.device,
            )
            torques = torques * self.torque_multi

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # ------------------------------------------------------------------ #
    # Doneness / rewards / observations                                  #
    # ------------------------------------------------------------------ #
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # gait command resampling + push curriculum (mirrors original
        # _post_physics_step_callback, which ran before termination checks)
        self._post_physics_step_callback()

        contact = self.contact_forces[:, self.penalised_contact_indices, :]
        terminated = torch.any(torch.norm(contact, dim=-1) > 1.0, dim=1)

        root_euler = get_euler_xyz_tensor(self.base_quat)
        terminated |= torch.abs(root_euler[:, 0]) > 1.5
        terminated |= torch.abs(root_euler[:, 1]) > 1.5

        time_out = self.episode_length_buf > self.max_episode_length
        # fresh time-out signal every step (used by PPO for bootstrapping)
        self.extras["time_outs"] = time_out
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        # mirrors LeggedRobot.compute_reward (only_positive + termination after clip)
        rew_buf = torch.zeros(self.num_envs, device=self.device)
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            rew_buf = torch.clip(rew_buf, min=0.0)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            rew_buf += rew
            self.episode_sums["termination"] += rew
        return rew_buf

    def _get_observations(self) -> dict:
        cfg = self.cfg
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_stance_mask()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.0

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1
        )

        base_quat = self.base_quat
        base_euler_xyz = get_euler_xyz_tensor(base_quat)
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])

        # critic (no lag) - 73 dims
        diff = self.dof_pos - self.ref_dof_pos
        privileged_obs_buf = torch.cat((
            self.command_input,                                                  # 5
            (self.dof_pos - self.default_joint_pd_target) * cfg.obs_scales.dof_pos,  # 12
            self.dof_vel * cfg.obs_scales.dof_vel,                                # 12
            self.actions,                                                        # 12
            diff,                                                                # 12
            base_lin_vel * cfg.obs_scales.lin_vel,                                # 3
            base_ang_vel * cfg.obs_scales.ang_vel,                                # 3
            base_euler_xyz * cfg.obs_scales.quat,                                 # 3
            self.rand_push_force[:, :2],                                          # 2
            self.rand_push_torque,                                                # 3
            self.env_frictions[:, 0],                                             # 1
            self.body_mass / 10.0,                                                # 1
            stance_mask,                                                          # 2
            contact_mask,                                                         # 2
        ), dim=-1)

        # dof lag observation
        dr = cfg.domain_rand
        if dr.add_dof_lag:
            idx = torch.arange(self.num_envs, device=self.device)
            lag_t = self.dof_lag_timestep.long()
            self.lagged_dof_pos = self.dof_lag_buffer[idx, : self.num_actions, lag_t]
            self.lagged_dof_vel = self.dof_lag_buffer[idx, -self.num_actions:, lag_t]
        elif dr.add_dof_pos_vel_lag:
            idx = torch.arange(self.num_envs, device=self.device)
            self.lagged_dof_pos = self.dof_pos_lag_buffer[idx, :, self.dof_pos_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_vel_lag_buffer[idx, :, self.dof_vel_lag_timestep.long()]
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

        if dr.add_imu_lag:
            idx = torch.arange(self.num_envs, device=self.device)
            self.lagged_imu = self.imu_lag_buffer[idx, :, self.imu_lag_timestep.long()]
            self.lagged_base_ang_vel = self.lagged_imu[:, :3].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:, -3:].clone()
        else:
            self.lagged_base_ang_vel = base_ang_vel[:, :3]
            self.lagged_base_euler_xyz = base_euler_xyz[:, -3:]

        q = (self.lagged_dof_pos - self.default_dof_pos) * cfg.obs_scales.dof_pos
        dq = self.lagged_dof_vel * cfg.obs_scales.dof_vel

        obs_buf = torch.cat((
            self.command_input,                              # 5
            q,                                               # 12
            dq,                                              # 12
            self.actions,                                    # 12
            self.lagged_base_ang_vel * cfg.obs_scales.ang_vel,  # 3
            self.lagged_base_euler_xyz * cfg.obs_scales.quat,   # 3
        ), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() + (
                2 * torch.rand_like(obs_buf) - 1
            ) * self.noise_scale_vec * cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_buf)

        obs_buf_all = torch.stack(
            [self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1
        )  # (N, T, K)
        clip_obs = cfg.normalization.clip_observations
        obs_policy = obs_buf_all.reshape(self.num_envs, -1)
        obs_policy = torch.clip(obs_policy, -clip_obs, clip_obs)
        obs_critic = torch.clip(
            torch.cat([self.critic_history[i] for i in range(cfg.env.c_frame_stack)], dim=1),
            -clip_obs, clip_obs,
        )

        # update last-step buffers for reward terms (mirrors original post_physics_step)
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]
        self.last_contact_forces[:] = self.contact_forces[:]
        self.last_torques[:] = self.torques[:]

        return {"policy": obs_policy, "critic": obs_critic}

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(cfg.env.num_single_obs, device=self.device)
        noise_scales = cfg.noise.noise_scales
        na = self.num_actions
        nc = cfg.env.num_commands
        noise_vec[0:nc] = 0.0
        noise_vec[nc: nc + na] = noise_scales.dof_pos * cfg.obs_scales.dof_pos
        noise_vec[nc + na: nc + 2 * na] = noise_scales.dof_vel * cfg.obs_scales.dof_vel
        noise_vec[nc + 2 * na: nc + 3 * na] = 0.0
        noise_vec[nc + 3 * na: nc + 3 * na + 3] = noise_scales.ang_vel * cfg.obs_scales.ang_vel
        noise_vec[nc + 3 * na + 3: nc + 3 * na + 6] = noise_scales.quat * cfg.obs_scales.quat
        return noise_vec

    # ------------------------------------------------------------------ #
    # Reset                                                              #
    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        cfg = self.cfg

        # scene reset restores default root/joint state for these envs
        super()._reset_idx(env_ids)

        # command curriculum (common across envs)
        if cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # resample randomized properties for the reset envs
        # NOTE: mass/link-mass/friction randomization is applied ONCE at init
        # (mirrors the original pipeline, which set rigid body properties at
        # env creation and only re-randomized DOF props on reset)
        self.randomize_dof_props(env_ids)
        self.randomize_lag_props(env_ids)

        # dof state: default + small noise, zero velocity
        dof_pos = self.default_dof_pos.expand(len(env_ids), -1) + torch_rand_float(
            -0.1, 0.1, (len(env_ids), self.num_dof), device=self.device
        )
        dof_vel = torch.zeros(len(env_ids), self.num_dof, device=self.device)
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # root state: env origin (+ random xy offset inside terrain tile) at init height
        # note: write API expects quaternion in wxyz -> identity is (w=1) at index 3
        env_origins = self._terrain.env_origins[env_ids]
        root_state = torch.zeros(len(env_ids), 13, device=self.device)
        root_state[:, 3] = 1.0  # w = 1 (wxyz identity)
        root_state[:, :3] = env_origins + torch.tensor(
            [0.0, 0.0, cfg.env.init_state_z], device=self.device
        )
        root_state[:, :2] += torch_rand_float(
            -cfg.terrain.terrain_length / 2, cfg.terrain.terrain_length / 2,
            (len(env_ids), 2), device=self.device,
        )
        self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # armature randomization (solver side)
        if cfg.domain_rand.randomize_joint_armature:
            self._robot.write_joint_armature_to_sim(self.joint_armatures[env_ids], env_ids=env_ids)
        if cfg.domain_rand.randomize_joint_friction:
            self._robot.write_joint_friction_coefficient_to_sim(
                self.joint_friction_coeffs[env_ids], env_ids=env_ids
            )
        if cfg.domain_rand.randomize_joint_damping:
            self._robot.write_joint_damping_to_sim(self.joint_damping_coeffs[env_ids], env_ids=env_ids)

        # reset history buffers
        self.last_last_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_rigid_state[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_root_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.phase_length_buf[env_ids] = 0
        self.gait_start[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device) * 0.5

        self.generate_gait_time(env_ids)
        self._resample_commands_gait(env_ids)

        # episode bookkeeping
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        if cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(cfg.env.c_frame_stack):
            self.critic_history[i][env_ids] *= 0

    # ------------------------------------------------------------------ #
    # Domain randomization sampling                                      #
    # ------------------------------------------------------------------ #
    def randomize_dof_props(self, env_ids):
        dr = self.cfg.domain_rand
        if dr.randomize_torque:
            lo, hi = dr.torque_multiplier_range
            self.torque_multi[env_ids] = torch_rand_float(lo, hi, (len(env_ids), self.num_actions), device=self.device)
        if dr.randomize_motor_offset:
            lo, hi = dr.motor_offset_range
            self.motor_offsets[env_ids] = torch_rand_float(lo, hi, (len(env_ids), self.num_actions), device=self.device)
        if dr.randomize_gains:
            lo, hi = dr.stiffness_multiplier_range
            self.randomized_p_gains[env_ids] = (
                torch_rand_float(lo, hi, (len(env_ids), self.num_actions), device=self.device) * self.p_gains
            )
            lo, hi = dr.damping_multiplier_range
            self.randomized_d_gains[env_ids] = (
                torch_rand_float(lo, hi, (len(env_ids), self.num_actions), device=self.device) * self.d_gains
            )
        if dr.randomize_coulomb_friction:
            lo, hi = dr.joint_coulomb_range
            self.randomized_joint_coulomb[env_ids] = torch_rand_float(lo, hi, (len(env_ids), self.num_actions), device=self.device)
            lo, hi = dr.joint_viscous_range
            self.randomized_joint_viscous[env_ids] = torch_rand_float(lo, hi, (len(env_ids), self.num_actions), device=self.device)
        if dr.randomize_joint_armature:
            lo, hi = dr.joint_armature_range
            self.joint_armatures[env_ids] = torch_rand_float(lo, hi, (len(env_ids), self.num_actions), device=self.device)

    def randomize_rigid_body_props(self, env_ids):
        """One-shot rigid body randomization (called once after init).

        Mirrors the original pipeline: base/link masses are sampled per env at
        creation time (fixed for the whole training), applied through PhysX
        link masses with inertia recomputation. env_frictions are likewise
        sampled once and exposed through the privileged observations.
        """
        dr = self.cfg.domain_rand
        if dr.randomize_base_mass:
            lo, hi = dr.added_mass_range
            self.payload_masses[env_ids] = torch_rand_float(lo, hi, (len(env_ids), 1), device=self.device)
        if dr.randomize_link_mass:
            lo, hi = dr.added_link_mass_range
            self.link_masses[env_ids] = torch_rand_float(lo, hi, (len(env_ids), self.num_bodies), device=self.device)
        if dr.randomize_com:
            ranges = dr.com_displacement_range
            self.com_displacements[env_ids, :] = torch.cat(
                (
                    torch_rand_float(ranges[0][0], ranges[0][1], (len(env_ids), 1), device=self.device),
                    torch_rand_float(ranges[1][0], ranges[1][1], (len(env_ids), 1), device=self.device),
                    torch_rand_float(ranges[2][0], ranges[2][1], (len(env_ids), 1), device=self.device),
                ),
                dim=-1,
            )

        # apply link mass randomization through PhysX link masses (CPU tensors)
        if dr.randomize_base_mass or dr.randomize_link_mass:
            masses = torch.as_tensor(self._robot.root_physx_view.get_masses())  # (num_envs, num_bodies) CPU
            default_masses = self._robot.data.default_mass.clone().cpu()
            masses[env_ids] = default_masses[env_ids]
            if dr.randomize_link_mass:
                masses[env_ids, 1:] = masses[env_ids, 1:] * self.link_masses[env_ids, 1:].cpu()
            if dr.randomize_base_mass:
                masses[env_ids, 0] = masses[env_ids, 0] + self.payload_masses[env_ids, 0].cpu()
            # body mass exposed to the critic = merged base-link mass (original semantics)
            self.body_mass[env_ids] = masses[env_ids, 0].to(self.device)
            self._robot.root_physx_view.set_masses(masses, env_ids.cpu())

            # recompute inertias from the mass ratios (original used recomputeInertia=True)
            ratios = masses[env_ids] / default_masses[env_ids]
            inertias = torch.as_tensor(self._robot.root_physx_view.get_inertias())  # (N, B, 9)
            default_inertias = self._robot.data.default_inertia.clone().cpu()
            inertias[env_ids] = default_inertias[env_ids] * ratios.unsqueeze(-1)
            self._robot.root_physx_view.set_inertias(inertias, env_ids.cpu())

    def randomize_lag_props(self, env_ids):
        dr = self.cfg.domain_rand
        if dr.add_lag:
            self.lag_buffer[env_ids, :, :] = 0.0
            if dr.randomize_lag_timesteps:
                self.lag_timestep[env_ids] = torch.randint(
                    dr.lag_timesteps_range[0], dr.lag_timesteps_range[1] + 1,
                    (len(env_ids),), device=self.device,
                )
        if dr.add_dof_lag:
            self.dof_lag_buffer[env_ids, :, :] = 0.0
            if dr.randomize_dof_lag_timesteps:
                self.dof_lag_timestep[env_ids] = torch.randint(
                    dr.dof_lag_timesteps_range[0], dr.dof_lag_timesteps_range[1] + 1,
                    (len(env_ids),), device=self.device,
                )

    # ------------------------------------------------------------------ #
    # Command / gait scheduling (ported verbatim)                        #
    # ------------------------------------------------------------------ #
    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        if self.cfg.commands.sw_switch:
            stand_command = (
                torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold
            )
            self.phase_length_buf[stand_command] = 0
            phase = (self.phase_length_buf * self.step_dt / cycle_time + self.gait_start) * (~stand_command)
        else:
            phase = self.episode_length_buf * self.step_dt / cycle_time + self.gait_start
        return phase

    def _get_stance_mask(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = (sin_pos >= 0).float()
        stance_mask[:, 1] = (sin_pos < 0).float()
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        return stance_mask

    def generate_gait_time(self, envs):
        if len(envs) == 0:
            return
        random_tensor_list = []
        for i in range(len(self.cfg.commands.gait)):
            name = self.cfg.commands.gait[i]
            gait_time_range = self.cfg.commands.gait_time_range[name]
            random_tensor_single = torch_rand_float(
                gait_time_range[0], gait_time_range[1], (len(envs), 1), device=self.device
            )
            random_tensor_list.append(random_tensor_single)
        random_tensor = torch.cat(random_tensor_list, dim=1)
        current_sum = torch.sum(random_tensor, dim=1, keepdim=True)
        scaled_tensor = random_tensor * (self.max_episode_length / current_sum)
        scaled_tensor[:, 1:] = scaled_tensor[:, :-1].clone()
        scaled_tensor[:, 0] *= 0.0
        self.gait_time[envs] = torch.cumsum(scaled_tensor, dim=1).int()

    def _resample_commands(self):
        for i in range(len(self.cfg.commands.gait)):
            env_ids = (self.episode_length_buf == self.gait_time[:, i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                name = "_resample_" + self.cfg.commands.gait[i] + "_command"
                resample_command = getattr(self, name)
                resample_command(env_ids)

    def _resample_stand_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_sagittal_command(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_lateral_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_rotate_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device,
            ).squeeze(1)

    def _resample_walk_omnidirectional_command(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device,
            ).squeeze(1)

    def _resample_commands_gait(self, env_ids):
        """On reset: immediately resample commands for the reset envs.

        In the original pipeline, episode_length_buf == 0 matches gait segment 0
        (which has zero length by construction), so the first gait type's
        resample function fires right away.
        """
        first_gait = self.cfg.commands.gait[0]
        getattr(self, "_resample_" + first_gait + "_command")(env_ids)

    def update_command_curriculum(self, env_ids):
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > (
            0.8 * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.25, -self.cfg.commands.max_curriculum / 2, 0.0
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0.0, self.cfg.commands.max_curriculum
            )

    # ------------------------------------------------------------------ #
    # Push robots (velocity impulse)                                     #
    # ------------------------------------------------------------------ #
    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )
        self.rand_push_torque[:] = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device
        )
        root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        root_vel[:, :2] = self.rand_push_force[:, :2]
        root_vel[:, 3:] = self.rand_push_torque
        self._robot.write_root_velocity_to_sim(root_vel)

    # ------------------------------------------------------------------ #
    # Post-physics callback (gait, pushes) - called from step override   #
    # ------------------------------------------------------------------ #
    def _post_physics_step_callback(self):
        self.phase_length_buf += 1
        self._resample_commands()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )
        if self.cfg.domain_rand.push_robots:
            i = int(self.common_step_counter / self.cfg.domain_rand.update_step)
            if i >= len(self.cfg.domain_rand.push_duration):
                i = len(self.cfg.domain_rand.push_duration) - 1
            duration = self.cfg.domain_rand.push_duration[i] / self.step_dt
            if self.common_step_counter % self.cfg.domain_rand.push_interval <= duration:
                self._push_robots()
            else:
                self.rand_push_force.zero_()
                self.rand_push_torque.zero_()

    # ------------------------------------------------------------------ #
    # Reference gait state                                               #
    # ------------------------------------------------------------------ #
    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()

        deltas = self.cfg.rewards.final_swing_joint_delta_pos
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        # left swing (sin < 0 branch in original indexing)
        sin_pos_l[sin_pos_l > 0] = 0
        for j in range(6):
            self.ref_dof_pos[:, j] = -sin_pos_l * deltas[j]
        # right
        sin_pos_r[sin_pos_r < 0] = 0
        for j in range(6, 12):
            self.ref_dof_pos[:, j] = sin_pos_r * deltas[j]

        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0.0
        self.ref_action = 2 * self.ref_dof_pos
        self.ref_dof_pos += self.default_dof_pos

    # ------------------------------------------------------------------ #
    # Reward machinery                                                   #
    # ------------------------------------------------------------------ #
    def _prepare_reward_function(self):
        scale_dict = {}
        for name in dir(self.cfg.rewards.scales):
            if name.startswith("_"):
                continue
            val = getattr(self.cfg.rewards.scales, name)
            if isinstance(val, (int, float)):
                scale_dict[name] = float(val)
        self.reward_scales = {k: v * self.step_dt for k, v in scale_dict.items() if v != 0}
        self.reward_functions = []
        self.reward_names = []
        for name, _ in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))
        self.episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self.reward_scales.keys()
        }

    def _reward_termination(self):
        return self.reset_buf.float() * ~self.reset_time_outs

    # ---------------------- reward terms (ported) ---------------------- #
    def _reward_ref_joint_pos(self):
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        stand_command = (
            torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold
        )
        pos_target[stand_command] = self.default_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        r[stand_command] = 1.0
        return r

    def _reward_feet_distance(self):
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.foot_min_dist
        max_df = self.cfg.rewards.foot_max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.foot_min_dist
        max_df = self.cfg.rewards.foot_max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew = rew * contact.float()
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) < 0.05] = 1
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask > 0), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * self.contact_filt
        self.feet_air_time += self.step_dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] = 1
        reward = torch.where(contact == (stance_mask > 0), 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        base_quat = self.base_quat
        base_euler_xyz = get_euler_xyz_tensor(base_quat)
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        quat_mismatch = torch.exp(-torch.sum(torch.abs(base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.0

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400),
            dim=1,
        )

    def _reward_default_joint_pos(self):
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, [1, 2, 5]]
        right_yaw_roll = joint_diff[:, [7, 8, 11]]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        stance_mask = self._get_stance_mask()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1
        ) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - self.cfg.rewards.feet_to_ankle_distance)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        return torch.exp(-torch.norm(root_acc, dim=1) * 3)

    def _reward_vel_mismatch_exp(self):
        base_quat = self.base_quat
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])
        lin_mismatch = torch.exp(-torch.square(base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(base_ang_vel[:, :2], dim=1) * 5.0)
        return (lin_mismatch + ang_mismatch) / 2.0

    def _reward_track_vel_hard(self):
        base_quat = self.base_quat
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])
        lin_vel_error = torch.norm(self.commands[:, :2] - base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)
        ang_vel_error = torch.abs(self.commands[:, 2] - base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error

    def _reward_tracking_lin_vel(self):
        base_quat = self.base_quat
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        stand_command = (
            torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold
        )
        err_sq = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        err_abs = torch.sum(torch.abs(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        r_square = torch.exp(-err_sq * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-err_abs * self.cfg.rewards.tracking_sigma * 2)
        return torch.where(stand_command, r_abs, r_square)

    def _reward_tracking_ang_vel(self):
        base_quat = self.base_quat
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])
        stand_command = (
            torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold
        )
        err_sq = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        err_abs = torch.abs(self.commands[:, 2] - base_ang_vel[:, 2])
        r_square = torch.exp(-err_sq * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-err_abs * self.cfg.rewards.tracking_sigma * 2)
        return torch.where(stand_command, r_abs, r_square)

    def _reward_feet_clearance(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        feet_z = self.rigid_state[:, self.feet_indices, 2] - self.cfg.rewards.feet_to_ankle_distance
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        swing_mask = 1 - self._get_stance_mask()
        rew_pos = (self.feet_height > self.cfg.rewards.target_feet_height) * (
            self.feet_height < self.cfg.rewards.target_feet_height_max
        )
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        base_quat = self.base_quat
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        absolute_speed = torch.abs(base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)
        sign_mismatch = torch.sign(base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])
        reward = torch.zeros_like(base_lin_vel[:, 0])
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.0
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.05).float()

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_ankle_torques(self):
        ankle_idx = [4, 5, 10, 11]
        return torch.sum(torch.square(self.torques[:, ankle_idx]), dim=1)

    def _reward_feet_rotation(self):
        feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        feet_euler_xyz = get_euler_xyz_tensor(feet_quat)
        rotation = torch.sum(torch.square(feet_euler_xyz[:, :, :2]), dim=[1, 2])
        return torch.exp(-rotation * 15)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.step_dt), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_action_smoothness(self):
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_stand_still(self):
        stand_command = (
            torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold
        )
        r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        return torch.where(stand_command, r.clone(), torch.zeros_like(r))

    def _reward_feet_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        ).float()

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_dof_torque_limits(self):
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0),
            dim=1,
        )
