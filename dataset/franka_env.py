#!/usr/bin/env python3
"""
FrankaTaskEnv: simple MuJoCo environment with two tasks:

1) Pick-and-place:
   - 1 cube (`cube_pick`) and a target region (`target_bowl`)
   - Randomize cube and/or target positions on the table
   - Success: cube center inside target region (and gripper open, if you wire it)

2) Stacking:
   - 2 cubes (`cube_A`, `cube_B`)
   - Randomize both cubes on the table (non-overlapping)
   - Success: cube A stacked on cube B within XY/Z tolerances

Includes:
   - env.reset(task="pick_place" or "stack")
   - env.set_task(...)
   - env.step(action) -> obs, reward, done, info
   - step_count, done, success_flag, episode_reward bookkeeping
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

import mujoco
import numpy as np


TaskName = Literal["pick_place", "stack"]


class Task(enum.Enum):
    PICK_PLACE = "pick_place"
    STACK = "stack"


@dataclass
class StepResult:
    obs: Dict[str, np.ndarray]
    reward: float
    done: bool
    info: Dict[str, Any]


class FrankaTaskEnv:
    def __init__(
        self,
        xml_path: str = "franka_emika_panda.xml",
        rng_seed: int = 0,
        max_steps: int = 300,
    ) -> None:
        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        

        # Simple RGB renderer (used for trajectory logging / replay)
        # Defaults can be overridden when calling render().
        self._renderer = mujoco.Renderer(self.model, width=640, height=480)
        # Try to locate a default overhead camera if it exists
        try:
            self._overhead_cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead"
            )
        except Exception:
            self._overhead_cam_id = -1

        # RNG and episode settings
        self.rng = np.random.default_rng(rng_seed)
        self.max_steps = max_steps
        
        # Randomization control (can be set before reset)
        self.randomize_red_cube = True  # Default: randomize red cube
        self.randomize_blue_cube = True  # Default: keep blue cube fixed

        # Cache body ids
        self.cube_pick_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_pick"
        )
        self.cube_A_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_A"
        )
        self.cube_B_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_B"
        )
        self.target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_bowl"
        )

        # End-effector body (Panda hand) for simple task-space control
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )

        # Table region for randomization (x, y) – tune as needed
        self.table_x_min, self.table_x_max = 0.35, 0.65
        self.table_y_min, self.table_y_max = -0.15, 0.15
        self.table_z_cube = 0.45  # cube center at table height

        # Target region radius (for pick-and-place success)
        self.target_radius = 0.05

        # Stacking tolerances
        # Expected vertical gap between cube centers ~ 2 * cube_half_size = 0.06
        self.stack_xy_tol = 0.025
        self.stack_z_center = 0.06
        self.stack_z_tol = 0.02

        # Episode state
        self.task: Task = Task.PICK_PLACE
        self.step_count: int = 0
        self.episode_reward: float = 0.0
        self.success_flag: bool = False

        # Cache a "home" configuration - same as franka_move.py
        mujoco.mj_resetData(self.model, self.data)
        # Home joint positions (7 arm joints)
        self.home_q = np.array([
            0.0,
            0.0,
            0.0,
            -1.57079,
            0.0,
            1.57079,
            -0.7853,
        ])
        self.home_qpos = self.data.qpos.copy()
        # Set arm joints to home position
        self.home_qpos[:7] = self.home_q

    # ------------------------------------------------------------------
    # Task API
    # ------------------------------------------------------------------

    def set_task(self, task: TaskName) -> None:
        self.task = Task(task)

    def reset(self, task: Optional[TaskName] = None) -> Dict[str, np.ndarray]:
        """
        Reset environment for a new episode.

        If `task` is provided, switch to that task first.
        """
        if task is not None:
            self.set_task(task)

        # Episode bookkeeping
        self.step_count = 0
        self.episode_reward = 0.0
        self.success_flag = False

        # Reset MuJoCo internal state
        mujoco.mj_resetData(self.model, self.data)
        
        # Restore home configuration (same initialization as franka_move.py)
        # Position control: desired position = home_q
        self.data.qpos[:7] = self.home_q
        self.data.qvel[:7] = 0.0
        self.data.ctrl[:7] = self.home_q
        
        # Gripper open (ctrl range 0–255 from XML)
        act_dim = self.data.ctrl.shape[0]
        if act_dim > 7:
            self.data.ctrl[7] = 255.0  # Open gripper
        
        # Recompute derived quantities after setting state
        mujoco.mj_forward(self.model, self.data)

        # Task-specific randomization
        if self.task is Task.PICK_PLACE:
            self._randomize_pick_place(
                randomize_red=self.randomize_red_cube,
                randomize_blue=self.randomize_blue_cube
            )
        elif self.task is Task.STACK:
            self._randomize_stack()

        # Let physics settle so cubes rest on the table
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs()

    # ------------------------------------------------------------------
    # Randomization helpers
    # ------------------------------------------------------------------

    def _randomize_pick_place(self, randomize_red: bool = True, randomize_blue: bool = True) -> None:
        """
        Randomize cube positions: ±0.1 in x and y around base positions.
        Ensures no collision between cubes (minimum separation 0.08m).
        
        Args:
            randomize_red: If True, randomize red cube (cube_pick) position
            randomize_blue: If True, randomize blue cube (cube_A) position
        """
        # Base positions from XML
        red_base_x, red_base_y = 0.7, -0.1  # cube_pick
        blue_base_x, blue_base_y = 0.7, 0.1  # cube_A
        
        # First, set fixed cubes to their base positions
        if not randomize_red and self.cube_pick_id >= 0:
            # Keep red at base position
            self._set_body_pos(self.cube_pick_id, [red_base_x, red_base_y, self.table_z_cube])
            red_pos = np.array([red_base_x, red_base_y, self.table_z_cube])
        elif self.cube_pick_id >= 0:
            # Will randomize red, initialize to base for now
            red_pos = np.array([red_base_x, red_base_y, self.table_z_cube])
        else:
            red_pos = np.array([red_base_x, red_base_y, self.table_z_cube])
            
        if not randomize_blue and self.cube_A_id >= 0:
            # Keep blue at base position
            self._set_body_pos(self.cube_A_id, [blue_base_x, blue_base_y, self.table_z_cube])
            blue_pos = np.array([blue_base_x, blue_base_y, self.table_z_cube])
        elif self.cube_A_id >= 0:
            # Will randomize blue, initialize to base for now
            blue_pos = np.array([blue_base_x, blue_base_y, self.table_z_cube])
        else:
            blue_pos = np.array([blue_base_x, blue_base_y, self.table_z_cube])
        
        # Randomize red cube if requested
        if randomize_red and self.cube_pick_id >= 0:
            max_tries = 50
            for _ in range(max_tries):
                x = self.rng.uniform(red_base_x - 0.1, red_base_x + 0.1)
                y = self.rng.uniform(red_base_y - 0.1, red_base_y + 0.1)
                # Check collision with blue cube (if blue is fixed)
                if not randomize_blue and self.cube_A_id >= 0:
                    dist = np.linalg.norm([x - blue_pos[0], y - blue_pos[1]])
                    if dist > 0.08:  # Minimum separation (cube size ~0.06, add margin)
                        self._set_body_pos(self.cube_pick_id, [x, y, self.table_z_cube])
                        red_pos = np.array([x, y, self.table_z_cube])  # Update for blue check
                        break
                else:
                    # Blue is also being randomized or doesn't exist, so just set red
                    self._set_body_pos(self.cube_pick_id, [x, y, self.table_z_cube])
                    red_pos = np.array([x, y, self.table_z_cube])
                    break
        
        # Randomize blue cube if requested (after red is set)
        if randomize_blue and self.cube_A_id >= 0:
            max_tries = 50
            for _ in range(max_tries):
                x = self.rng.uniform(blue_base_x - 0.1, blue_base_x + 0.1)
                y = self.rng.uniform(blue_base_y - 0.1, blue_base_y + 0.1)
                # Check collision with red cube (red is now fixed or already randomized)
                # Get current red position
                if self.cube_pick_id >= 0:
                    current_red_pos = self.data.xpos[self.cube_pick_id].copy()
                else:
                    current_red_pos = red_pos
                dist = np.linalg.norm([x - current_red_pos[0], y - current_red_pos[1]])
                if dist > 0.08:  # Minimum separation
                    self._set_body_pos(self.cube_A_id, [x, y, self.table_z_cube])
                    break

        # Don't randomize target for now (only moving above cube)
        # if self.target_id >= 0:
        #     tx = self.rng.uniform(self.table_x_min, self.table_x_max)
        #     ty = self.rng.uniform(self.table_y_min, self.table_y_max)
        #     # target sits on table top (height ~0.42)
        #     self._set_body_pos(self.target_id, [tx, ty, 0.42])

    def _randomize_stack(self) -> None:
        """Randomize cube_A and cube_B on the table, avoiding overlaps."""
        if self.cube_A_id < 0 or self.cube_B_id < 0:
            return

        # Sample non-overlapping positions
        max_tries = 50
        for _ in range(max_tries):
            xA = self.rng.uniform(self.table_x_min, self.table_x_max)
            yA = self.rng.uniform(self.table_y_min, self.table_y_max)
            xB = self.rng.uniform(self.table_x_min, self.table_x_max)
            yB = self.rng.uniform(self.table_y_min, self.table_y_max)
            if np.linalg.norm([xA - xB, yA - yB]) > 0.08:
                break

        self._set_body_pos(self.cube_A_id, [xA, yA, self.table_z_cube])
        self._set_body_pos(self.cube_B_id, [xB, yB, self.table_z_cube])

    def _set_body_pos(self, body_id: int, xyz) -> None:
        """
        Set body position in qpos for a body with a freejoint.

        Uses body_jntadr -> joint id -> jnt_qposadr to find the correct
        qpos slice. Assumes the first joint for this body is a freejoint.
        """
        if body_id < 0:
            return

        # Index of the first joint belonging to this body
        jnt_adr = int(self.model.body_jntadr[body_id])
        if jnt_adr < 0:
            return  # body has no joint

        # qpos index for this joint
        qpos_adr = int(self.model.jnt_qposadr[jnt_adr])

        # For a freejoint, the first 3 qpos entries are the Cartesian position
        self.data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(xyz, dtype=float)
        # Leave orientation (last 4 entries) unchanged

    # ------------------------------------------------------------------
    # Step / reward / success
    # ------------------------------------------------------------------

    def step(self, ctrl: Optional[np.ndarray] = None) -> StepResult:
        """
        Apply control (if provided), step the simulation once,
        and compute reward / done / info.
        """
        if ctrl is not None:
            self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()
        success = self._check_success()
        reward = 1.0 if success and not self.success_flag else 0.0
        self.episode_reward += reward

        # Lock in success once achieved
        self.success_flag = self.success_flag or success

        done = self.step_count >= self.max_steps or self.success_flag

        info = {
            "task": self.task.value,
            "step_count": self.step_count,
            "success": self.success_flag,
            "episode_reward": self.episode_reward,
        }

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def _check_success(self) -> bool:
        if self.task is Task.PICK_PLACE:
            return self._is_success_pick_place()
        elif self.task is Task.STACK:
            return self._is_success_stack()
        return False

    def _is_success_pick_place(self) -> bool:
        """Success when end-effector is within tolerance of target (cube position + 0.2m in z)."""
        if self.cube_pick_id < 0 or self.ee_body_id < 0:
            return False

        cube_pos = self.data.xpos[self.cube_pick_id]
        ee_pos = self.data.xpos[self.ee_body_id]
        
        # Target position: same x, y as cube, z is 0.2m above cube
        target_pos = cube_pos.copy()
        target_pos[2] = cube_pos[2] + 0.2
        
        # Check if end-effector is within tolerance
        # x and y within 0.05m, z within 0.1m
        x_diff = abs(ee_pos[0] - target_pos[0])
        y_diff = abs(ee_pos[1] - target_pos[1])
        z_diff = abs(ee_pos[2] - target_pos[2])
        
        return x_diff <= 0.1 and y_diff <= 0.1 and z_diff <= 0.1

    def _is_success_stack(self) -> bool:
        """
        Cube A stacked on cube B:
        - XY centers within tolerance
        - Vertical center distance ~ stack_z_center with some tolerance
        """
        if self.cube_A_id < 0 or self.cube_B_id < 0:
            return False

        pos_A = self.data.xpos[self.cube_A_id]
        pos_B = self.data.xpos[self.cube_B_id]

        dist_xy = np.linalg.norm(pos_A[:2] - pos_B[:2])
        dz = pos_A[2] - pos_B[2]

        return bool(
            dist_xy <= self.stack_xy_tol
            and abs(dz - self.stack_z_center) <= self.stack_z_tol
        )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Simple observation: positions of cubes and target (if present)."""
        obs: Dict[str, np.ndarray] = {}

        if self.cube_pick_id >= 0:
            obs["cube_pick_pos"] = self.data.xpos[self.cube_pick_id].copy()
        if self.cube_A_id >= 0:
            obs["cube_A_pos"] = self.data.xpos[self.cube_A_id].copy()
        if self.cube_B_id >= 0:
            obs["cube_B_pos"] = self.data.xpos[self.cube_B_id].copy()
        if self.target_id >= 0:
            obs["target_pos"] = self.data.xpos[self.target_id].copy()

        return obs

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def render(
        self,
        width: int = 640,
        height: int = 480,
        camera_name: Optional[str] = "overhead",
    ) -> np.ndarray:
        """
        Render an RGB frame from the specified camera.

        Returns an array of shape (H, W, 3) uint8.
        """
        # Recreate renderer if resolution has changed
        if (
            self._renderer.width != width
            or self._renderer.height != height
        ):
            self._renderer = mujoco.Renderer(self.model, width=width, height=height)

        cam_id = -1
        if camera_name is not None:
            try:
                cam_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                )
            except Exception:
                cam_id = -1

        if cam_id >= 0:
            # Use a fixed camera from the model
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = cam_id
            self._renderer.update_scene(self.data, camera=camera)
        else:
            # Default free camera
            self._renderer.update_scene(self.data)

        rgb = self._renderer.render()
        return rgb

    def get_full_state(self) -> Dict[str, np.ndarray]:
        """
        Convenience helper for logging:

        Returns a dict with:
          - qpos, qvel
          - object poses we care about (cube/target positions)
        """
        state: Dict[str, np.ndarray] = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }
        obs = self._get_obs()
        # Flatten object poses into a single array for supervision
        # Only include objects that actually exist in the XML:
        # - cube_pick_pos (red cube)
        # - cube_A_pos (blue cube)
        # Note: cube_B and target_bowl are not in the current XML
        obj_keys = ["cube_pick_pos", "cube_A_pos"]
        obj_list = [obs[k] for k in obj_keys if k in obs]
        if obj_list:
            state["obj_poses"] = np.stack(obj_list, axis=0)  # (N, 3) where N = number of existing objects
        else:
            state["obj_poses"] = np.zeros((0, 3), dtype=float)
        return state


# ----------------------------------------------------------------------
# Tiny demo loops
# ----------------------------------------------------------------------

if __name__ == "__main__":
    env = FrankaTaskEnv()

    # --- Demo: pick-and-place episode ---
    print("=== Pick-and-place episode ===")
    obs = env.reset(task="pick_place")
    print("Initial obs:", {k: v.round(3) for k, v in obs.items()})

    done = False
    while not done:
        # No control for now – just let cubes and robot sit
        step = env.step()
        done = step.done

    print("Done:", step.info)

    # --- Demo: stacking episode ---
    print("\n=== Stacking episode ===")
    obs = env.reset(task="stack")
    print("Initial obs:", {k: v.round(3) for k, v in obs.items()})

    done = False
    while not done:
        step = env.step()
        done = step.done

    print("Done:", step.info)


