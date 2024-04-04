# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math

import numpy as np
import torch
from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.maths import tensor_clamp
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.panda import Panda
from omniisaacgymenvs.robots.articulations.views.panda_view import PandaView
import omni.isaac.core.utils.stage as stage_utils
from pxr import Usd, UsdGeom
import omni.isaac.core.utils.stage as stage_utils
from pxr import Usd, UsdGeom



class PandaBlock(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.dt = 1 / 60.0

        self._num_actions = 9
        self._num_observations = self._num_actions + 7

        RLTask.__init__(self, name, env)
        return


    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")),
            self._device,
        )
        lfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")),
            self._device,
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")),
            self._device,
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self._num_envs, 1))
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)


    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.num_props = self._task_cfg["env"]["numProps"]

    def set_up_scene(self, scene) -> None:
        self.get_panda()
        self.get_props()


        super().set_up_scene(scene, filter_collisions=False)


        # Add panda view to the scene
        self._pandas = PandaView(prim_paths_expr="/World/envs/.*/panda", name="panda_view")
        scene.add(self._pandas)
        scene.add(self._pandas._hands)
        scene.add(self._pandas._lfingers)
        scene.add(self._pandas._rfingers)


        # Add props view to the scene
        self._props = RigidPrimView(prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False)
        scene.add(self._props)

        return

    def get_panda(self):
        # NOTE: Basically self.default_zero_env_path is /World/envs/env_0
        panda = Panda(prim_path=self.default_zero_env_path + "/panda", name="panda")
        self._sim_config.apply_articulation_settings(
            "panda", get_prim_at_path(panda.prim_path), self._sim_config.parse_actor_config("panda")
        )

    def get_props(self):
        prop = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/prop/prop_0",
            name="prop",
            translation=torch.tensor([0.2, 0.0, 0.1]),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            color=torch.tensor([0.2, 0.4, 0.6]),
            size=0.08,
            density=100.0,
        )
        self._sim_config.apply_articulation_settings(
            "prop", get_prim_at_path(prop.prim_path), self._sim_config.parse_actor_config("prop")
        )

    def get_observations(self) -> dict:
        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._pandas._hands.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos
        end_effector_orientations = end_effector_orientations[:, [3, 0, 1, 2]]

        # Get dof positions
        dof_pos = self._pandas.get_joint_positions(clone=False)

        self.obs_buf[..., 0:self._num_actions] = dof_pos
        self.obs_buf[..., self._num_actions:self._num_actions + 3] = end_effector_positions
        self.obs_buf[..., self._num_actions + 3:self._num_actions + 7] = end_effector_orientations

        observations = {self._pandas.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Calculate targte pose
        self.actions = actions.clone().to(self._device)
        targets = self.panda_dof_targets + self.dt * self.actions * self.action_scale

        # Clamp action
        self.panda_dof_targets[:] = tensor_clamp(targets, self.panda_dof_lower_limits, self.panda_dof_upper_limits)

        # Set target pose
        env_ids_int32 = torch.arange(self._pandas.count, dtype=torch.int32, device=self._device)
        self._pandas.set_joint_position_targets(self.panda_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        # Reset DOF states for robots in selected envs
        self._pandas.set_joint_position_targets(self.initial_dof_positions, indices=env_ids_32)
        self._pandas.set_joint_positions(self.initial_dof_positions, indices=env_ids_32)
        self._pandas.set_joint_velocities(self.initial_dof_velocities, indices=env_ids_32)

        # Reset root state for robots in selected envs
        self._pandas.set_world_poses(
            self.initial_robot_pos[env_ids_64],
            self.initial_robot_rot[env_ids_64],
            indices=env_ids_32
        )

        # reset props
        self._props.set_world_poses(
            self.initial_prop_pos[env_ids_64],
            self.initial_prop_rot[env_ids_64],
            indices=env_ids_32
        )

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.num_panda_dofs = self._pandas.num_dof
        self.panda_dof_pos = torch.zeros((self._num_envs, self.num_panda_dofs), device=self._device)

        dof_limits = self._pandas.get_dof_limits()
        self.panda_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.panda_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.panda_dof_targets = torch.zeros(
            (self._num_envs, self.num_panda_dofs), dtype=torch.float, device=self._device
        )

        self.initial_robot_pos, self.initial_robot_rot = self._pandas.get_world_poses()
        self.initial_dof_positions = self._pandas.get_joint_positions()
        self.initial_dof_velocities = torch.zeros_like(self.initial_dof_positions, device=self._device)

        self.initial_prop_pos, self.initial_prop_rot = self._props.get_world_poses()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # TODO: change
        self.rew_buf[:] = 1

        # distance from hand to the drawer
        # d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        # dist_reward = 1.0 / (1.0 + d**2)
        # dist_reward *= dist_reward
        # dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    def is_done(self) -> None:
        # TODO: change
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )
