from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
import omni.isaac.core.utils.stage as stage_utils


class PR2View(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "PR2View",
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
            prim_paths_expr="/World/envs/.*/pr2/r_gripper_tool_frame",
            name="hands_view",
            reset_xform_properties=False
        )
        # Add left finger as RigidPrimView
        self._lfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/pr2/r_gripper_l_finger_tip_link",
            name="lfingers_view",
            reset_xform_properties=False
        )
        # Add right finger as RigidPrimView
        self._rfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/pr2/r_gripper_r_finger_tip_link",
            name="rfingers_view",
            reset_xform_properties=False,
        )
        print('view end =========================')

    def initialize(self, physics_sim_view):
        print('============================================== initialize gripper indices 1')
        print(physics_sim_view)
        super().initialize(physics_sim_view)
        print('============================================== initialize gripper indices 2')
        self._gripper_indices = [self.get_dof_index("l_gripper_joint"), self.get_dof_index("r_gripper_joint")]
        print(self._gripper_indices)

    @property
    def gripper_indices(self):
        return self._gripper_indices
