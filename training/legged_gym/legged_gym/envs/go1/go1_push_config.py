# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.legged_robot_push_box_config import LeggedRobotPushCfg
import numpy as np

class Go1PushBoxCfg(LeggedRobotPushCfg):
    class env(LeggedRobotPushCfg.env):
        num_observations = 49+6 #box_rel_xy(2), target_rel_from_box_xy(2), heading_err(1), box_speed_toward_target(1)]
        num_envs = 1280
        episode_length_s = 20
        send_timeouts = True

    class init_state( LeggedRobotPushCfg.init_state ):
        pos = [0.0, 0.0, 0.37] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,   # [rad]
            'RL_hip_joint': 0.,   # [rad]
            'FR_hip_joint': 0.,  # [rad]
            'RR_hip_joint': 0.,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3
        class ranges:
            pos_rho = [1.0, 3.0] # min max [m]
            pos_theta = [-3.14, 3.14]  # rad
            heading = [-3.14, 3.14]



    class control( LeggedRobotPushCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.65}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4



    class asset( LeggedRobotPushCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"] # collision reward
        terminate_after_contacts_on = ["base"] # termination rewrad
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        load_dynamic_object = False
        object_files = {
        # '{LEGGED_GYM_ROOT_DIR}/resources/objects/DiningChair/model.urdf': 0.4,
        # '{LEGGED_GYM_ROOT_DIR}/resources/objects/OfficeChair/model.urdf': 0.4,
        # '{LEGGED_GYM_ROOT_DIR}/resources/objects/cylindar.urdf': 0.4,
        }
        # object_num = 8
        # test_mode = False
        # test_obj_pos = [] # to be overwritten with a 3d tensor


    class terrain( LeggedRobotPushCfg.terrain ):
        terrain_types = "plane"  # do not duplicate!
        # terrain_proportions = [0.5, 0.5, 0.5]
        curriculum = False # 这里为了test  采用了plane add curriculum = False
        measure_heights = False

    class domain_rand:
        randomize_friction = True
        friction_range = [-0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5]
        randomize_dof_bias = True
        max_dof_bias = 0.08
        randomize_timer_minus = 2.0  # timer_left is initialized with randomization: U(T-this, T)

        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0.0  # not used
        
        randomize_yaw = True
        init_yaw_range = [-3.14, 3.14]
        randomize_roll = False
        randomize_pitch = False
        randomize_xy = True
        init_x_range = [-0.5, 0.5]
        init_y_range = [-0.5, 0.5]
        randomize_velo = True
        init_vlinx_range = [-0.5,0.5]
        init_vliny_range = [-0.5,0.5]
        init_vlinz_range = [-0.5,0.5]
        init_vang_range = [-0.5,0.5]
        randomize_init_dof = True
        init_dof_factor=[0.5, 1.5]
        stand_bias3 = [0.0, 0.0, 0.0]

        erfi = True
        erfi_torq_lim = 7.0/9  # per level, curriculum

    class sensors:
        class ray2d:
            enable = False
            log2 = False
            min_dist = 0.1
            max_dist = 6.0
            theta_start = - np.pi/4
            theta_end = np.pi/4 + 0.0001
            theta_step = np.pi/20
            x_0 = -0.05
            y_0 = 0.0
            front_rear = False
            illusion = True  # add illusion when there is noise
            raycolor = (0,0.5,0.5)
        
        class depth_cam:
            enable = False
            resolution = [1280//8,720//8]
            x = 0.0
            y = 0
            z = 0.27
            far_plane = 10.0
            hfov = 102.0
            min_ = 0.1
            max_ = 6.0        

    class box:
        size = [0.4, 0.4, 0.4]       # (lx, ly, lz) [m]
        mass = 1.0                  # [kg]
        friction = 1.0               # 箱子接触摩擦
        restitution = 0.0
        spawn_distance_front = 2.0   # 相对机器人前向 2m
        spawn_y_jitter = 0.1         # 轻微左右随机
        max_lin_vel = 1.0
        max_ang_vel = 1.0
    class target:
        # 目标点设置为「箱子初始点沿世界x正方向再前进2m」
        extra_forward = 2.0
        reach_threshold = 0.3

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.2
            height_measurements = 2.0
            ray2d = 1.0
        clip_observations = 100.
        clip_actions = 100.


    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            ray2d = 0.2  # 2^0.2 = 1.1487

    class rewards():
        class scales():
            termination = -100. 
            reach_pos_target_soft = 60.0
            reach_pos_target_tight = 60.0
            reach_heading_target = 30.0
            reach_pos_target_times_heading = 0.0
            velo_dir = 10.0
            torques = -0.0005
            dof_pos_limits = -20.0
            dof_vel = -0.0005
            torque_limits = -20.0
            dof_vel_limits = -20.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.0e-7
            collision = -100.
            feet_collision = -100.
            action_rate = -0.01
            stand_still_pos = -10.0
            orientation = -20.0
            fly = -20.0
            nomove = -20.0
            # 正向
            progress_box = 20.0          # 箱子距目标的“距离减少/步长”
            contact_box = 0.2            # 与箱子保持接触
            box_speed = 1.5          # 箱子朝向目标的速度
            align_heading = 0.5      # 机器人朝向目标对齐
            behind_box = 0.5         # 机器人在箱子后方（沿目标方向的后方）
            # 代价
            torques = -0.0005
            dof_vel = -0.0005
            action_rate = -0.01
            box_side_slip = -0.3     # 箱子侧向速度惩罚
            # 终止
            termination = -100.0

        soft_dof_pos_limit = 0.95
        base_height_target = 0.25
        only_positive_rewards = False
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.5
        heading_target_sigma = 1.0
        rew_duration = 2.0
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.85
        max_contact_force = 100.



class Go1PushRoughCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.003
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go1_push_box_rough'
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48 # per iteration
        max_iterations = 20000 # number of policy updates
        # logging
        save_interval = 200 # check for potential saves every this many iterations
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
