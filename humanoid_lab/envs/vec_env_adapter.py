# Copyright (c) 2024, AgiBot Inc. All rights reserved.
"""Adapts the IsaacLab DirectRLEnv to the rsl_rl-style VecEnv interface used
by DHOnPolicyRunner (step -> 5-tuple with privileged obs)."""

import torch


class VecEnvAdapter:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self._last_obs = None

    # ---- dims ----
    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def num_obs(self):
        return self.cfg.env.num_observations

    @property
    def num_privileged_obs(self):
        return self.cfg.env.num_privileged_obs

    @property
    def num_actions(self):
        return self.cfg.env.num_actions

    @property
    def num_short_obs(self):
        return self.cfg.env.short_frame_stack * self.cfg.env.num_single_obs

    @property
    def num_single_obs(self):
        return self.cfg.env.num_single_obs

    @property
    def device(self):
        return self.env.device

    # obs-noise annealing passthrough (must be an explicit property: plain
    # attribute assignment on the adapter would shadow the env attribute)
    @property
    def noise_level_factor(self):
        return self.env.noise_level_factor

    @noise_level_factor.setter
    def noise_level_factor(self, value):
        self.env.noise_level_factor = value

    @property
    def max_episode_length(self):
        return self.env.max_episode_length

    def __getattr__(self, name):
        # transparent passthrough (episode_length_buf, max_episode_length_s, ...)
        return getattr(self.env, name)

    # ---- vec env api ----
    def step(self, actions):
        obs, rew, terminated, time_outs, extras = self.env.step(actions)
        self._last_obs = obs
        dones = terminated | time_outs
        return obs["policy"], obs["critic"], rew, dones, extras

    def reset(self):
        obs, extras = self.env.reset()
        self._last_obs = obs
        return obs["policy"], obs["critic"]

    def get_observations(self) -> torch.Tensor:
        return self._last_obs["policy"]

    def get_privileged_observations(self):
        return self._last_obs["critic"]
