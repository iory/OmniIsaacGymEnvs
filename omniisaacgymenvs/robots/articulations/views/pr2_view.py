from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class PR2View(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "PR2View",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        # By adding it as a RigidPrimView, it is possible to get the Pose using functions such as get_world_pose()
        # Add hand as RigidPrimView
        self._hands = RigidPrimView(
            prim_paths_expr="/World/envs/.*/pr2/base_link",
            name="hands_view",
            reset_xform_properties=False
        )
        # Add left finger as RigidPrimView
        self._lfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/pr2/r_gripper_l_finger_tip_frame",
            name="lfingers_view",
            reset_xform_properties=False
        )
        # Add right finger as RigidPrimView
        self._rfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/pr2/l_gripper_l_finger_tip_frame",
            name="rfingers_view",
            reset_xform_properties=False,
        )

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
        self._gripper_indices = [self.get_dof_index("l_gripper_joint"), self.get_dof_index("r_gripper_joint")]

    @property
    def gripper_indices(self):
        return self._gripper_indices
