# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. 
# SPDX-License-Identifier: BSD-3-Clause
#
# Push-Box Env in the same "LeggedRobotPos" style:
# - Spawns a box 2m in front of the robot
# - Target is placed further ahead of the box
# - Rewards encourage pushing the box to the target

from time import time
import os
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.utils.math import quat_apply, yaw_quat, wrap_to_pi

# from .legged_robot_config import LeggedRobotCfg
from .go1_push_config import Go1PushBoxCfg


class Go1RobotPushBox(LeggedRobot):
    """A push-box task implemented in the same structure as LeggedRobotPos."""
    cfg: Go1PushBoxCfg  # expects cfg with fields: env, domain_rand, rewards, and custom: box, target
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        """Add task-specific buffers alongside base buffers."""
        super()._init_buffers()

        # track box-target distance for progress shaping
        self.prev_box_target_dist = torch.zeros(self.num_envs, device=self.device)
        # success flag (for logging)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # observation scales for the added features (keep consistent with base)
        self.add_noise = self.cfg.noise.add_noise
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # a small helper tensor for drawing/heading
        self.forward_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.forward_vec[:, 0] = 1.0



    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        if self.cfg.domain_rand.randomize_dof_bias:
            self.dof_bias[env_ids] = self.dof_bias[env_ids].uniform_(-self.cfg.domain_rand.max_dof_bias, self.cfg.domain_rand.max_dof_bias)
        if self.cfg.domain_rand.erfi:
            self.erfi_rnd[env_ids] = self.erfi_rnd[env_ids].uniform_(0., 1.)
        if self.cfg.asset.load_dynamic_object:
            self.obj_state_rand[env_ids] = self.obj_state_rand[env_ids].uniform_(0., 1.)
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.contact_filt[env_ids] = False
        self.last_contacts[env_ids] = False
        # reset timer
        self.timer_left[env_ids] = -self.cfg.domain_rand.randomize_timer_minus * torch.rand(len(env_ids), device=self.device) + self.cfg.env.episode_length_s
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _create_envs(self):
        """Extend base env creation by spawning one dynamic box per env and computing target poses."""
        super()._create_envs()

        # ---- create box asset ----
        bx, by, bz = getattr(self.cfg.box, "size", [0.4, 0.4, 0.4])
        box_opts = gymapi.AssetOptions()
        box_opts.fix_base_link = False
        self.box_asset = self.gym.create_box(self.sim, bx, by, bz, box_opts)

        # shape properties (friction, restitution)
        self._box_shape_props = gymapi.ShapeProperties()
        self._box_shape_props.friction = getattr(self.cfg.box, "friction", 1.0)
        self._box_shape_props.restitution = getattr(self.cfg.box, "restitution", 0.0)

        # body properties (mass)
        self._box_mass = getattr(self.cfg.box, "mass", 8.0)

        self.box_handles = []
        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)

        # Spawn one box per env and remember target position
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            origin = self.env_origins[i]

            spawn_x = origin[0].item() + getattr(self.cfg.box, "spawn_distance_front", 2.0)
            y_jitter = getattr(self.cfg.box, "spawn_y_jitter", 0.1)
            spawn_y = origin[1].item() + (np.random.rand() * 2 - 1) * y_jitter
            spawn_z = origin[2].item() + bz * 0.5

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(spawn_x, spawn_y, spawn_z)

            box_h = self.gym.create_actor(env_ptr, self.box_asset, pose, f"box_{i}", i, 0)
            self.box_handles.append(box_h)

            # set shape properties (friction, etc.)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, box_h)
            for s in shape_props:
                s.friction = self._box_shape_props.friction
                s.restitution = self._box_shape_props.restitution
            self.gym.set_actor_rigid_shape_properties(env_ptr, box_h, shape_props)

            # set body mass
            body_props = self.gym.get_actor_rigid_body_properties(env_ptr, box_h)
            for b in body_props:
                b.mass = self._box_mass
            self.gym.set_actor_rigid_body_properties(env_ptr, box_h, body_props, True)

            # target is further forward in +x from box
            extra_f = getattr(self.cfg.target, "extra_forward", 2.0)
            self.target_positions[i, 0] = pose.p.x + extra_f
            self.target_positions[i, 1] = pose.p.y
            self.target_positions[i, 2] = pose.p.z

        # rigid body state tensor (robot + box)
        self.rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # find each env's box rigid body index in the big tensor
        self._cache_box_rb_indices()

    def _cache_box_rb_indices(self):
        """Cache the sim-domain rigid body tensor index for each box."""
        self.box_rb_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            # The box has only one body (its base). Index 0 in actor domain -> map to sim domain:
            rb_idx = self.gym.find_actor_rigid_body_index(env_ptr, self.box_handles[i], 0, gymapi.DOMAIN_SIM)
            self.box_rb_indices[i] = rb_idx



    # -------------- Resets & Episode Management --------------

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # curriculum update if any
        if getattr(self.cfg.terrain, "curriculum", False):
            self._update_terrain_curriculum(env_ids)

        # reset robot (dofs & root)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # randomize robot-side buffers (bias, etc.)
        if getattr(self.cfg.domain_rand, "randomize_dof_bias", False):
            self.dof_bias[env_ids] = self.dof_bias[env_ids].uniform_(
                -self.cfg.domain_rand.max_dof_bias, self.cfg.domain_rand.max_dof_bias
            )
        if getattr(self.cfg.domain_rand, "erfi", False):
            self.erfi_rnd[env_ids] = self.erfi_rnd[env_ids].uniform_(0.0, 1.0)

        # reset box poses (front 2m) and recompute target
        bx, by, bz = getattr(self.cfg.box, "size", [0.4, 0.4, 0.4])
        for i in env_ids.tolist():
            env_ptr = self.envs[i]
            origin = self.env_origins[i]

            spawn_x = origin[0].item() + getattr(self.cfg.box, "spawn_distance_front", 2.0)
            y_jitter = getattr(self.cfg.box, "spawn_y_jitter", 0.1)
            spawn_y = origin[1].item() + (np.random.rand() * 2 - 1) * y_jitter
            spawn_z = origin[2].item() + bz * 0.5

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(spawn_x, spawn_y, spawn_z)
            self.gym.set_actor_transform(env_ptr, self.box_handles[i], pose)

            extra_f = getattr(self.cfg.target, "extra_forward", 2.0)
            self.target_positions[i, 0] = pose.p.x + extra_f
            self.target_positions[i, 1] = pose.p.y
            self.target_positions[i, 2] = pose.p.z

        # clear episode buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.contact_filt[env_ids] = False
        self.last_contacts[env_ids] = False
        self.success_buf[env_ids] = False

        # initialize progress distance
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_pos = self.rb_states[self.box_rb_indices, 0:3]
        tgt_pos = self.target_positions
        self.prev_box_target_dist[env_ids] = torch.norm((box_pos - tgt_pos)[env_ids, :2], dim=-1)

        # fill extras logging
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        if getattr(self.cfg.env, "send_timeouts", False):
            self.extras["time_outs"] = self.time_out_buf

    def _update_terrain_curriculum(self, env_ids):
        # optional: you can base curriculum on success or stability
        pass

    # -------------- Step Callbacks --------------

    def _post_physics_step_callback(self):
        """Update contact-latched mask, push robots if enabled, etc."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def check_termination(self):
        """Set reset flags for terminal conditions."""
        # base collisions (as in LeggedRobot)
        died = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)

        # success: box reaches target
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_pos = self.rb_states[self.box_rb_indices, 0:3]
        dist_now = torch.norm((box_pos - self.target_positions)[:, :2], dim=-1)
        success = dist_now < getattr(self.cfg.target, "reach_threshold", 0.3)
        self.success_buf = success | self.success_buf

        # timeout
        time_out = (self.episode_length_buf * self.dt) >= self.cfg.env.episode_length_s

        self.time_out_buf = time_out
        self.reset_buf = died | time_out | success

    # -------------- Observations --------------

    def _get_noise_scale_vec(self, cfg):
        """Match the style of LeggedRobotPos for noise scaling, with appended added features."""
        noise_vec = torch.empty_like(self.obs_buf[0])
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        # Base 49 dims (contact4, ang_vel3, gravity3, cmd3, dof_pos12, dof_vel12, last actions12)
        noise_vec[:4] = 0.0
        noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity * noise_level
        noise_vec[10:13] = 0.0
        noise_vec[13:25] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[25:37] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[37:49] = 0.0

        # Added 6 dims: [rel_box_body(2), tgt_rel_from_box(2), heading_err(1), box_speed_toward_target(1)]
        added = torch.zeros(6, device=self.device)
        noise_vec = torch.cat([noise_vec, added], dim=0)
        return noise_vec

    def compute_observations(self):
        """Base obs + 6D push-box features (same concatenation style as LeggedRobotPos)."""
        # ----- base obs (49) -----
        self.obs_buf = torch.cat((
            self.contact_filt.float() * 2 - 1.0,                               # 0:4
            self.base_ang_vel * self.obs_scales.ang_vel,                        # 4:7
            self.projected_gravity,                                             # 7:10
            self.commands[:, :3],                                               # 10:13
            (self.dof_pos - self.default_dof_pos - self.dof_bias) * self.obs_scales.dof_pos,  # 13:25
            self.dof_vel * self.obs_scales.dof_vel,                             # 25:37
            self.actions                                                        # 37:49
        ), dim=-1)

        # ----- push-box features (+6) -----
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_pos = self.rb_states[self.box_rb_indices, 0:3]
        box_vel = self.rb_states[self.box_rb_indices, 7:10]

        # box relative position in BODY frame (yaw-only inverse)
        rel_box_world = box_pos[:, :2] - self.root_states[:, :2]
        rel_box_body3 = quat_rotate_inverse(yaw_quat(self.base_quat[:]),
                                            torch.cat([rel_box_world, torch.zeros_like(rel_box_world[:, :1])], dim=-1))
        rel_box_body = rel_box_body3[:, :2]

        # target relative from box (world 2D)
        tgt_rel_from_box = (self.target_positions - box_pos)[:, :2]

        # heading error: face the box
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        desired_heading = torch.atan2(rel_box_world[:, 1], rel_box_world[:, 0])
        heading_err = wrap_to_pi(desired_heading - heading).unsqueeze(1)

        # box speed toward target
        tgt_dir_world = torch.nn.functional.normalize(tgt_rel_from_box, dim=-1)
        box_speed_toward_target = (box_vel[:, :2] * tgt_dir_world).sum(dim=-1, keepdim=True)

        extras = torch.cat([rel_box_body, tgt_rel_from_box, heading_err, box_speed_toward_target], dim=-1)  # 6D
        self.obs_buf = torch.cat((self.obs_buf, extras), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # -------------- Debug Draw --------------

    def _draw_debug_vis(self):
        super()._draw_debug_vis()
        # draw target spheres at each env
        sphere_geom = gymutil.WireframeSphereGeometry(0.08, 8, 8, None, color=(0.1, 0.9, 0.1))
        for i in range(self.num_envs):
            x, y, z = self.target_positions[i].tolist()
            pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    # -------------- Reward Terms (LeggedGym style) --------------

    # def _command_duration_mask(self, duration):
    #     # optional mask similar to LeggedRobotPos; here keep as 1
    #     return torch.ones(self.num_envs, device=self.device)

    def _reward_progress_box(self):
        """Distance reduction per step: (prev - now)/dt."""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_pos = self.rb_states[self.box_rb_indices, 0:3]
        dist_now = torch.norm((box_pos - self.target_positions)[:, :2], dim=-1)
        progress = (self.prev_box_target_dist - dist_now) / (self.dt + 1e-6)
        # clip for stability
        progress = torch.clamp(progress, -1.0, 1.0)
        self.prev_box_target_dist = dist_now.clone()
        return progress

    def _reward_contact_box(self):
        """Proximity proxy for 'contact' – higher when robot is near the box."""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_pos = self.rb_states[self.box_rb_indices, 0:3]
        robot_box_dist = torch.norm((self.root_states[:, :2] - box_pos[:, :2]), dim=-1)
        # exponential falloff
        return torch.exp(-5.0 * robot_box_dist)

    def _reward_box_speed(self):
        """Velocity of the box along the target direction."""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_pos = self.rb_states[self.box_rb_indices, 0:3]
        box_vel = self.rb_states[self.box_rb_indices, 7:10]
        tgt_dir = torch.nn.functional.normalize((self.target_positions - box_pos)[:, :2], dim=-1)
        return (box_vel[:, :2] * tgt_dir).sum(dim=-1)

    def _reward_align_heading(self):
        """Cosine of heading error between robot forward and direction to box."""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_pos = self.rb_states[self.box_rb_indices, 0:3]
        rel = box_pos[:, :2] - self.root_states[:, :2]
        desired_heading = torch.atan2(rel[:, 1], rel[:, 0])

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        err = torch.abs(wrap_to_pi(desired_heading - heading))
        return torch.cos(err)  # in [-1, 1]

    def _reward_behind_box(self):
        """Encourage being behind the box along the target direction (negative projection)."""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_pos = self.rb_states[self.box_rb_indices, 0:3]
        tgt_dir = torch.nn.functional.normalize((self.target_positions - box_pos)[:, :2], dim=-1)
        robot_rel = (self.root_states[:, :2] - box_pos[:, :2])
        proj = (robot_rel * tgt_dir).sum(dim=-1)  # >0 front, <0 behind
        return torch.clamp(-proj, min=0.0)

    def _reward_box_side_slip(self):
        """Penalize lateral slip of the box wrt target direction."""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_pos = self.rb_states[self.box_rb_indices, 0:3]
        box_vel = self.rb_states[self.box_rb_indices, 7:10]
        tgt_dir = torch.nn.functional.normalize((self.target_positions - box_pos)[:, :2], dim=-1)
        ortho = torch.stack([-tgt_dir[:, 1], tgt_dir[:, 0]], dim=-1)
        side_v = (box_vel[:, :2] * ortho).sum(dim=-1)
        return torch.square(side_v)

    def _reward_termination(self):
        """Negative term on termination except timeouts or success (keep same signature)."""
        # compute in check_termination; here just build a mask
        died = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        time_out = (self.episode_length_buf * self.dt) >= self.cfg.env.episode_length_s
        # success mask
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_pos = self.rb_states[self.box_rb_indices, 0:3]
        dist_now = torch.norm((box_pos - self.target_positions)[:, :2], dim=-1)
        success = dist_now < getattr(self.cfg.target, "reach_threshold", 0.3)
        # penalize only hard failures (died) and not success/time_out
        return (died & (~time_out) & (~success)).float()
