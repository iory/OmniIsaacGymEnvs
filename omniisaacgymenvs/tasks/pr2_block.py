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
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.maths import tensor_clamp
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.pr2 import PR2
from omniisaacgymenvs.robots.articulations.views.pr2_view import PR2View
import omni.isaac.core.utils.stage as stage_utils
from pxr import Usd, UsdGeom
import omni.isaac.core.utils.stage as stage_utils



class PR2Block(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        print(stage_utils.print_stage_prim_paths())
        self.update_config(sim_config)

        self.dt = 1 / 60.0
        self._num_observations = 19
        self._num_actions = 12

        RLTask.__init__(self, name, env)
        return

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
        self.get_p2r()
        self.get_props()

        print('==================== setupscene ==============')
        print(stage_utils.print_stage_prim_paths())

        super().set_up_scene(scene, filter_collisions=False)

        print('==================== setupscene ============== 1')
        print(stage_utils.print_stage_prim_paths())

        # Add p2r view to the scene
        self._p2rs = PR2View(prim_paths_expr="/World/envs/.*/pr2", name="p2r_view")
        scene.add(self._p2rs)
        scene.add(self._p2rs._hands)
        scene.add(self._p2rs._lfingers)
        scene.add(self._p2rs._rfingers)

        print('==================== setupscene ============== 2')
        print(stage_utils.print_stage_prim_paths())

        # Add props view to the scene
        self._props = RigidPrimView(prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False)
        scene.add(self._props)
        print('==================== setupscene ============== 3')
        print(stage_utils.print_stage_prim_paths())
        
        return

    def get_p2r(self):
        # NOTE: Basically self.default_zero_env_path is /World/envs/env_0
        print('================================================= get pr2')
        p2r = PR2(prim_path=self.default_zero_env_path + "/pr2", name="p2r")
        self._sim_config.apply_articulation_settings(
            "p2r", get_prim_at_path(p2r.prim_path), self._sim_config.parse_actor_config("p2r")
        )

    def get_props(self):
        print('================================================= get props')
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
        print('================================================= get observations')
        end_effector_positions, end_effector_orientations = self._p2rs._hands.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos
        end_effector_orientations = end_effector_orientations[:, [3, 0, 1, 2]]

        # Get dof positions
        dof_pos = self._p2rs.get_joint_positions(clone=False)

        self.obs_buf[..., 0:12] = dof_pos
        self.obs_buf[..., 12:15] = end_effector_positions
        self.obs_buf[..., 15:19] = end_effector_orientations

        observations = {self._p2rs.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        print('================================================= pre physics step')
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Calculate targte pose
        self.actions = actions.clone().to(self._device)
        targets = self.p2r_dof_targets + self.dt * self.actions * self.action_scale

        # Clamp action
        self.p2r_dof_targets[:] = tensor_clamp(targets, self.p2r_dof_lower_limits, self.p2r_dof_upper_limits)

        # Set target pose
        env_ids_int32 = torch.arange(self._p2rs.count, dtype=torch.int32, device=self._device)
        self._p2rs.set_joint_position_targets(self.p2r_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        print('================================================= reset idx')
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        # Reset DOF states for robots in selected envs
        self._p2rs.set_joint_position_targets(self.initial_dof_positions, indices=env_ids_32)
        self._p2rs.set_joint_positions(self.initial_dof_positions, indices=env_ids_32)
        self._p2rs.set_joint_velocities(self.initial_dof_velocities, indices=env_ids_32)

        # Reset root state for robots in selected envs
        self._p2rs.set_world_poses(
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
        self.num_p2r_dofs = self._p2rs.num_dof
        self.p2r_dof_pos = torch.zeros((self._num_envs, self.num_p2r_dofs), device=self._device)
        
        dof_limits = self._p2rs.get_dof_limits()
        self.p2r_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.p2r_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.p2r_dof_targets = torch.zeros(
            (self._num_envs, self.num_p2r_dofs), dtype=torch.float, device=self._device
        )

        self.initial_robot_pos, self.initial_robot_rot = self._p2rs.get_world_poses()
        self.initial_dof_positions = self._p2rs.get_joint_positions()
        self.initial_dof_velocities = torch.zeros_like(self.initial_dof_positions, device=self._device)

        self.initial_prop_pos, self.initial_prop_rot = self._props.get_world_poses()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # TODO: change
        self.rew_buf[:] = 1

    def is_done(self) -> None:
        # TODO: change
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )
