from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
import omni.isaac.core.utils.stage as stage_utils


class PandaView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "PandaView",
    ) -> None:
        """[summary]"""
        print('==================================')
        print('====================================================')
        print('.............................')
        print(stage_utils.print_stage_prim_paths())
        print('>>>>>>>>>>>>>>>>>>>>>>>.')
        print(prim_paths_expr)

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        # By adding it as a RigidPrimView, it is possible to get the Pose using functions such as get_world_pose()
        # Add hand as RigidPrimView
        print('.............................')
        print(stage_utils.print_stage_prim_paths())
        self._hands = RigidPrimView(
            prim_paths_expr="/World/envs/.*/panda/panda_hand",
            name="hands_view",
            reset_xform_properties=False
        )
        # Add left finger as RigidPrimView
        self._lfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/panda/panda_leftfinger",
            name="lfingers_view",
            reset_xform_properties=False
        )
        # Add right finger as RigidPrimView
        self._rfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/panda/panda_rightfinger",
            name="rfingers_view",
            reset_xform_properties=False,
        )
        print('view end =========================')

    def initialize(self, physics_sim_view):
        print('============================================== initialize gripper indices 1')
        print(physics_sim_view)
        super().initialize(physics_sim_view)
        # print('============================================== initialize gripper indices 2')
        self._gripper_indices = [self.get_dof_index("panda_finger_joint1"), self.get_dof_index("panda_finger_joint2")]
        print(self._gripper_indices)

    @property
    def gripper_indices(self):
        return self._gripper_indices
