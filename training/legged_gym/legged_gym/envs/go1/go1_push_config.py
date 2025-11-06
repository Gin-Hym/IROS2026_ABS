from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO, LeggedRobotCfgPPOLagrangian
from legged_gym.envs.base.legged_robot_pos_config import LeggedRobotPosCfg
import numpy as np

class Go1PushBoxCfg(LeggedRobotPosCfg):
    class env(LeggedRobotPosCfg.env):
        num_observations = 49+6 #box_rel_xy(2), target_rel_from_box_xy(2), heading_err(1), box_speed_toward_target(1)]
        num_envs = 1280
        episode_length_s = 20
        send_timeouts = True


    class init_state( LeggedRobotPosCfg.init_state ):
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

    class control( LeggedRobotPosCfg.control ):
        control_type = 'P'
        stiffness = {'joint': 30.}
        damping = {'joint': 0.65}
        action_scale = 0.25
        decimation = 4

    class init_state:
        pos = [0.0, 0.0, 0.35]
        # 复用你默认关节中性位；如需要可加入 default_joint_angles

    class commands:
        curriculum = False


    class asset( LeggedRobotPosCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"] # collision reward
        terminate_after_contacts_on = ["base"] # termination rewrad
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter



    class domain_rand:
        randomize_friction = True
        friction_range = [0.3, 1.0]
        randomize_roll = False
        randomize_pitch = False
        randomize_yaw = False
        randomize_xy = True
        init_x_range = [-0.2, 0.2]
        init_y_range = [-0.2, 0.2]
        randomize_init_dof = True
        init_dof_factor = [0.9, 1.1]
        stand_bias3 = [0.0, 0.2, -0.3]

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
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.2

    class rewards:
        # 正向
        progress = 20.0          # 箱子距目标的“距离减少/步长”
        contact = 0.2            # 与箱子保持接触
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

class Go1PushRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.003
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go1_push_box_rough'

