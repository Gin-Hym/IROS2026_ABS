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

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

def circle_ray_query(x0: torch.Tensor, y0: torch.Tensor, thetas: torch.Tensor, center_circle: torch.Tensor, radius: float, min_: float=0.1, max_:float = 3.0):
    """
    x0:(n,1), y0:(n,1), thetas: (n,t), center: (n,2)
    for each env, compute the distances for the ray from (x0, y0) at thetas to cross the circle
    return shape (n, t)
    """
    stheta = torch.sin(thetas) # (n,t)
    ctheta = torch.cos(thetas) # (n,t)
    xc = center_circle[:,0:1] # (n,1)
    yc = center_circle[:,1:2] # (n,1)
    d_c2line = torch.abs(stheta*xc - ctheta*yc - stheta*x0 + ctheta*y0)  #(n,t)
    d_c0_square = torch.square(xc-x0) + torch.square(yc-y0)
    d_0p = torch.sqrt(d_c0_square - torch.square(d_c2line))
    semi_arc = torch.sqrt(radius**2 - torch.square(d_c2line))
    raydist = torch.nan_to_num(d_0p - semi_arc, nan = max_).clip(min=min_, max=max_)
    check_dir = ctheta * (xc-x0) + stheta * (yc-y0)
    raydist = (check_dir > 0) * raydist + (check_dir<=0) * max_
    return raydist

def box_ray_query(x0, y0, thetas, box_center, box_size, box_yaw, min_=0.1, max_=3.0):
    """
    计算射线与 2D 旋转矩形 (Box) 的交点距离
    x0, y0: 机器人位置 (n, 1)
    thetas: 射线角度 (n, t)
    box_center: 箱子中心 (n, 2)
    box_size: 箱子半长宽 [half_x, half_y] (float or tensor)
    box_yaw: 箱子的旋转角 (n, 1)
    """
    # 1. 将射线转换到 Box 的局部坐标系 (Local Frame)
    # 也就是把 Box 旋转回正方向 (yaw=0)，把射线起点和方向做反向旋转
    dx = torch.cos(thetas)
    dy = torch.sin(thetas)
    
    # 相对位置 (机器人 - 箱子)
    rel_x = x0 - box_center[:, 0:1]
    rel_y = y0 - box_center[:, 1:2]
    
    # 旋转矩阵 (反向旋转 -box_yaw)
    c_yaw = torch.cos(-box_yaw)
    s_yaw = torch.sin(-box_yaw)
    
    # 旋转后的起点 (Local Origin)
    local_x0 = rel_x * c_yaw - rel_y * s_yaw
    local_y0 = rel_x * s_yaw + rel_y * c_yaw
    
    # 旋转后的方向 (Local Direction)
    local_dx = dx * c_yaw - dy * s_yaw
    local_dy = dx * s_yaw + dy * c_yaw
    
    # 2. Slab 算法 (AABB Intersection)
    # 加上微小量 eps 防止除以 0
    eps = 1e-6
    inv_dx = 1.0 / (local_dx + eps * torch.sign(local_dx))
    inv_dy = 1.0 / (local_dy + eps * torch.sign(local_dy))
    
    # 假设 box_size 是半长 (half_size)
    # 计算 X 轴方向的进入面和射出面
    t1x = (-box_size - local_x0) * inv_dx
    t2x = (box_size - local_x0) * inv_dx
    t_min_x = torch.minimum(t1x, t2x)
    t_max_x = torch.maximum(t1x, t2x)
    
    # 计算 Y 轴方向的进入面和射出面
    t1y = (-box_size - local_y0) * inv_dy
    t2y = (box_size - local_y0) * inv_dy
    t_min_y = torch.minimum(t1y, t2y)
    t_max_y = torch.maximum(t1y, t2y)
    
    # 3. 求交集
    # 射线进入 Box 的时间是 X和Y方向较晚进入的那个
    t_enter = torch.maximum(t_min_x, t_min_y)
    # 射线离开 Box 的时间是 X和Y方向较早离开的那个
    t_exit = torch.maximum(torch.minimum(t_max_x, t_max_y), torch.tensor(0.0, device=x0.device))
    
    # 4. 判断是否击中
    # 条件：t_exit >= t_enter (有重叠) 且 t_exit > 0 (在前方)
    hit = (t_exit >= t_enter) & (t_exit > 0)
    
    # t_enter 可能为负（如果机器人在箱子内部），此时距离应为 0 或 t_exit
    dist = torch.where(t_enter > 0, t_enter, torch.zeros_like(t_enter))
    
    # 如果没击中，设为最大距离
    final_dist = torch.where(hit, dist, torch.tensor(max_, device=x0.device))
    
    return final_dist.clip(min=min_, max=max_)


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    quat_yaw = quat.clone().view(-1, 4)
    qx = quat_yaw[:, 0]
    qy = quat_yaw[:, 1]
    qz = quat_yaw[:, 2]
    qw = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:, :2] = 0.0
    quat_yaw[:, 2] = torch.sin(yaw / 2)
    quat_yaw[:, 3] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower