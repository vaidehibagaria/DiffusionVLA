#!/usr/bin/env python3
"""
Data collection pipeline for Franka pick-and-place tasks with image observations.

This script collects demonstration trajectories for training vision-based diffusion policies.
Each trajectory contains RGB images, robot joint positions, and text task descriptions.

Output format:
  Each .npz file contains:
    {
      "obs": np.array(list_of_obs_dicts, dtype=object),  # length T
      "actions": np.array(actions),                      # length T x act_dim
      "task": "pick_place",
      "success": bool,
    }

  Each obs dict has keys:
    {
      "image": (320, 240, 3) uint8,  # RGB image from overhead camera
      "qpos": (7,) float32,          # 7-DOF arm joint positions
      "text": str,                    # Task description (e.g., "red cube", "blue cube")
    }

Usage:
  python collect_trajectories_image.py --collect --num_episodes 100
  python collect_trajectories_image.py --replay data/pick_place_episode_000.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import mujoco
import mujoco.viewer as viewer

from franka_env import FrankaTaskEnv, StepResult


# ---------------------------------------------------------------------------
# Task-space control utilities
# ---------------------------------------------------------------------------

def compute_orientation_error(target_rotmat: np.ndarray, current_rotmat: np.ndarray) -> np.ndarray:
    """
    Compute orientation error as axis-angle representation.
    
    This converts the rotation difference between target and current orientations
    into an axis-angle vector suitable for control.
    
    Args:
        target_rotmat: (3, 3) target rotation matrix
        current_rotmat: (3, 3) current rotation matrix
    
    Returns:
        (3,) axis-angle error vector
    """
    rot_err_mat = target_rotmat @ current_rotmat.T
    trace = np.trace(rot_err_mat)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    if angle < 1e-6:
        return np.zeros(3)
    
    axis = np.array([
        rot_err_mat[2, 1] - rot_err_mat[1, 2],
        rot_err_mat[0, 2] - rot_err_mat[2, 0],
        rot_err_mat[1, 0] - rot_err_mat[0, 1]
    ])
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        return angle * axis
    return np.zeros(3)


def compute_ik_control(
    env: FrankaTaskEnv,
    target_pos: np.ndarray,
    target_rotmat: np.ndarray,
    K_task_pos: float = 3.0,
    K_task_rot: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute joint-space control commands using 6D task-space inverse kinematics.
    
    Uses position and orientation Jacobians to map task-space errors to joint velocities.
    This enables end-effector control in both position and orientation.
    
    Args:
        env: FrankaTaskEnv instance
        target_pos: (3,) target end-effector position
        target_rotmat: (3, 3) target end-effector orientation matrix
        K_task_pos: Position control gain
        K_task_rot: Orientation control gain
    
    Returns:
        q_desired: (7,) desired joint positions
        pos_err: (3,) position error
        rot_err: (3,) orientation error (axis-angle)
    """
    arm_dofs = 7
    ee_pos = env.data.xpos[env.ee_body_id].copy()
    ee_rotmat = env.data.xmat[env.ee_body_id].copy().reshape(3, 3)
    qpos = env.data.qpos.copy()
    
    # Compute task-space errors
    pos_err = target_pos - ee_pos
    rot_err = compute_orientation_error(target_rotmat, ee_rotmat)
    
    # Get position and orientation Jacobians for end-effector
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))
    mujoco.mj_jacBody(env.model, env.data, jacp, jacr, env.ee_body_id)
    
    # Extract Jacobians for arm joints only (7 DOF)
    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
    arm_dof_ids = []
    for name in arm_joint_names:
        j_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_adr = env.model.jnt_dofadr[j_id]
        arm_dof_ids.append(dof_adr)
    
    J_pos = jacp[:, arm_dof_ids]  # (3, 7) position Jacobian
    J_rot = jacr[:, arm_dof_ids]  # (3, 7) orientation Jacobian
    J = np.vstack([J_pos, J_rot])  # (6, 7) combined Jacobian
    
    # Compute desired joint velocities using transpose (pseudo-inverse)
    scaled_err = np.concatenate([K_task_pos * pos_err, K_task_rot * rot_err])
    qdot_des = J.T @ scaled_err
    
    # Integrate to get desired joint positions
    qpos_arm = qpos[arm_dof_ids]
    q_desired = qpos_arm[:arm_dofs] + qdot_des
    
    return q_desired, pos_err, rot_err


def go_to_cube(env: FrankaTaskEnv, target_rotmat: np.ndarray, cube_id: int, height: float = 0.2) -> Tuple[bool, np.ndarray]:
    """
    Move end-effector to specified height above target cube.
    
    Maintains home orientation while positioning above the cube. Used as the
    first phase of the pick-and-place sequence.
    
    Args:
        env: FrankaTaskEnv instance
        target_rotmat: (3, 3) target end-effector orientation
        cube_id: Body ID of target cube (env.cube_pick_id or env.cube_A_id)
        height: Height above cube to position end-effector (meters)
    
    Returns:
        completed: True if target position reached within tolerance
        ctrl: (8,) control command (7 arm joints + 1 gripper)
    """
    cube_pos = env.data.xpos[cube_id].copy()
    target_pos = cube_pos + np.array([0.0, 0.0, height])
    
    q_desired, pos_err, rot_err = compute_ik_control(env, target_pos, target_rotmat)
    
    pos_tol = 0.01  # 1cm position tolerance
    completed = np.linalg.norm(pos_err) < pos_tol
    
    # Build control command
    ctrl = env.data.ctrl.copy()
    ctrl[:7] = q_desired
    gripper_idx = ctrl.shape[0] - 1 if ctrl.shape[0] > 7 else None
    if gripper_idx is not None:
        ctrl[gripper_idx] = 255.0  # Gripper open
    
    return completed, ctrl


def move_down(env: FrankaTaskEnv, target_rotmat: np.ndarray, cube_id: int, height: float = 0.1) -> Tuple[bool, np.ndarray]:
    """
    Move end-effector down to lower height above target cube.
    
    Second phase of pick-and-place: moves closer to cube for grasping.
    
    Args:
        env: FrankaTaskEnv instance
        target_rotmat: (3, 3) target end-effector orientation
        cube_id: Body ID of target cube
        height: Height above cube (meters)
    
    Returns:
        completed: True if target reached
        ctrl: (8,) control command
    """
    cube_pos = env.data.xpos[cube_id].copy()
    target_pos = cube_pos + np.array([0.0, 0.0, height])
    
    q_desired, pos_err, rot_err = compute_ik_control(env, target_pos, target_rotmat)
    
    pos_tol = 0.01
    rot_tol = 0.01
    completed = np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol
    
    ctrl = env.data.ctrl.copy()
    ctrl[:7] = q_desired
    gripper_idx = ctrl.shape[0] - 1 if ctrl.shape[0] > 7 else None
    if gripper_idx is not None:
        ctrl[gripper_idx] = 255.0  # Gripper open
    
    return completed, ctrl


# ---------------------------------------------------------------------------
# Scripted pick-and-place controller
# ---------------------------------------------------------------------------

def scripted_pick_and_place_episode(
    env: FrankaTaskEnv,
    max_steps: int = 2000,
    mj_viewer: viewer.Viewer | None = None,
    text_label: str = "red cube",
    cube_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute a complete pick-and-place episode using scripted control.
    
    This function runs a state machine that:
    1. Moves end-effector above target cube
    2. Moves down to grasp height
    3. Records observations (images, qpos, text) at each step
    
    The scripted controller uses 6D task-space IK to control end-effector
    position and orientation, making it robust to varying cube positions.
    
    Args:
        env: FrankaTaskEnv instance
        max_steps: Maximum steps before timeout
        mj_viewer: Optional MuJoCo viewer for visualization
        text_label: Text description of task (e.g., "red cube", "blue cube")
        cube_id: Body ID of target cube (defaults to red cube)
    
    Returns:
        episode: Dict with keys:
            - "obs": array of observation dicts (image, qpos, text)
            - "actions": (T, 8) array of control commands
            - "task": task name string
            - "success": bool indicating if task completed
    """
    if cube_id is None:
        cube_id = env.cube_pick_id  # Default to red cube
    
    env.reset(task="pick_place")
    
    # Get home orientation by temporarily setting to home configuration
    # This ensures consistent end-effector orientation throughout the episode
    home_q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    temp_qpos = env.data.qpos.copy()
    temp_qvel = env.data.qvel.copy()
    env.data.qpos[:7] = home_q
    env.data.qvel[:7] = 0.0
    mujoco.mj_forward(env.model, env.data)
    
    # Extract end-effector orientation at home position
    ee_rotmat = env.data.xmat[env.ee_body_id].copy().reshape(3, 3)
    target_ee_rotmat = ee_rotmat.copy()
    
    # Restore actual initial state
    env.data.qpos[:] = temp_qpos
    env.data.qvel[:] = temp_qvel
    mujoco.mj_forward(env.model, env.data)

    # Episode logging buffers
    obs_buffer: List[Dict[str, Any]] = []
    act_buffer: List[np.ndarray] = []

    # State machine for pick-and-place sequence
    state = 'go_to_cube'
    success = False
    step_count = 0
    done = False

    while step_count < max_steps and not done:
        # Capture observation BEFORE applying action (standard RL convention)
        state_before = env.get_full_state()
        rgb_before = env.render(width=320, height=240, camera_name="overhead")
        
        # Build observation dictionary
        # Note: text_label is constant throughout episode (all timesteps have same text)
        obs = {
            "image": rgb_before,  # (320, 240, 3) uint8 RGB image
            "qpos": state_before["qpos"][:7],  # (7,) arm joint positions
            "text": text_label,  # Task description string
        }
        obs_buffer.append(obs)
        
        # Compute control action based on current state
        ctrl = None
        
        if state == 'go_to_cube':
            # Phase 1: Move to position above cube
            completed, ctrl = go_to_cube(env, target_ee_rotmat, cube_id, height=0.3)
            if completed:
                state = 'move_down'
        
        elif state == 'move_down':
            # Phase 2: Move closer to cube
            completed, ctrl = move_down(env, target_ee_rotmat, cube_id, height=0.1)
            if completed:
                success = True
                done = True
        
        if ctrl is None:
            # Fallback: maintain current control
            ctrl = env.data.ctrl.copy()
        
        # Apply action to environment
        step_result: StepResult = env.step(ctrl=ctrl)

        # Sync viewer if provided
        if mj_viewer is not None:
            mj_viewer.sync()
        
        # Store action command (7 arm joints + 1 gripper = 8 dims)
        action_target = ctrl[:8].copy()
        act_buffer.append(action_target)

        step_count += 1

    # Package episode data
    episode = {
        "obs": np.array(obs_buffer, dtype=object),
        "actions": np.array(act_buffer, dtype=float),
        "task": env.task.value,
        "success": success,
    }
    return episode


# ---------------------------------------------------------------------------
# Data I/O utilities
# ---------------------------------------------------------------------------

def save_episode_npz(episode: Dict[str, Any], out_path: str) -> None:
    """
    Save episode data to compressed .npz file.
    
    Args:
        episode: Episode dictionary with obs, actions, task, success
        out_path: Output file path
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=episode["obs"],
        actions=episode["actions"],
        task=episode["task"],
        success=episode["success"],
    )
    print(f"[save_episode_npz] Saved episode to: {out_path}")


def load_episode_npz(path: str) -> Dict[str, Any]:
    """
    Load episode data from .npz file.
    
    Args:
        path: Path to .npz file
    
    Returns:
        episode: Episode dictionary
    """
    data = np.load(path, allow_pickle=True)
    episode = {
        "obs": data["obs"],
        "actions": data["actions"],
        "task": str(data["task"]),
        "success": bool(data["success"]),
    }
    # Handle legacy "language" field for backward compatibility
    if "language" in data:
        episode["language"] = str(data["language"])
    return episode


def replay_episode(
    env: FrankaTaskEnv,
    episode: Dict[str, Any],
    render_width: int = 640,
    render_height: int = 480,
) -> None:
    """
    Replay a recorded episode in the MuJoCo environment.
    
    Note: This replays actions but does not attempt to match exact object
    positions from the original episode. Useful for visualization and debugging.
    
    Args:
        env: FrankaTaskEnv instance
        episode: Episode dictionary loaded from .npz file
        render_width: Render image width
        render_height: Render image height
    """
    print("[replay_episode] Resetting environment for replay...")
    env.reset(task=str(episode["task"]))

    actions: np.ndarray = episode["actions"]
    
    # Actions are 7D (arm joints), need to add gripper control for 8D ctrl
    act_dim = env.data.ctrl.shape[0]
    gripper_idx = act_dim - 1 if act_dim > 7 else None

    with viewer.launch_passive(env.model, env.data) as v:
        for t, act in enumerate(actions):
            # Build control command: arm joints + gripper
            ctrl = np.zeros(act_dim)
            ctrl[:7] = act  # Arm joints
            if gripper_idx is not None:
                ctrl[gripper_idx] = 255.0  # Gripper open
            
            _ = env.step(ctrl=ctrl)
            _ = env.render(
                width=render_width,
                height=render_height,
                camera_name="overhead",
            )
            v.sync()
            if t % 20 == 0:
                print(f"[replay_episode] Step {t}/{len(actions)}")

    print("[replay_episode] Replay finished.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main CLI for data collection and episode replay.
    
    Examples:
        # Collect 100 episodes
        python collect_trajectories_image.py --collect --num_episodes 100
        
        # Replay a saved episode
        python collect_trajectories_image.py --replay data/pick_place_episode_000.npz
    """
    parser = argparse.ArgumentParser(
        description="Collect and replay Franka pick-and-place trajectories with image observations."
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect trajectories and save them to data/ directory.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to collect when --collect is set.",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default="",
        help="Path to a .npz file to load and replay.",
    )
    args = parser.parse_args()

    # Locate MuJoCo XML file
    script_dir = Path(__file__).parent.absolute()
    xml_path = script_dir / "franka_emika_panda.xml"
    if not xml_path.exists():
        repo_root = script_dir.parent
        xml_path = repo_root / "setup" / "franka_emika_panda.xml"
    
    if not xml_path.exists():
        raise FileNotFoundError(
            f"Could not find franka_emika_panda.xml. "
            f"Looked in: {script_dir} and {script_dir.parent / 'setup'}"
        )
    
    env = FrankaTaskEnv(xml_path=str(xml_path), max_steps=2000)
    
    if args.collect:
        # Initialize environment before collection
        env.reset(task="pick_place")
        
        # Setup output directory
        out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        out_dir = os.path.abspath(out_dir)
        print(f"[collect] Saving episodes to: {out_dir}")

        # Define text labels for different cube types
        # These labels are used as conditioning signals for the diffusion policy
        red_texts = ["red cube"]
        blue_texts = ["blue cube"]

        # Collect episodes with live visualization
        with viewer.launch_passive(env.model, env.data) as v:
            for i in range(args.num_episodes):
                print(f"\n[collect] Episode {i + 1}/{args.num_episodes}")
                
                # Alternate between red and blue cube targets
                # Both cubes are randomized each episode for better generalization
                if i < 50:
                    text_label = np.random.choice(red_texts)
                    cube_id = env.cube_pick_id
                else:
                    text_label = np.random.choice(blue_texts)
                    cube_id = env.cube_A_id
                
                # Randomize both cube positions for each episode
                # This helps the model generalize to different cube configurations
                env.randomize_red_cube = True
                env.randomize_blue_cube = True
                
                print(f"[collect] Target: {text_label} (cube_id={cube_id})")
                print(f"[collect] Randomization: red={env.randomize_red_cube}, blue={env.randomize_blue_cube}")
                
                # Execute scripted episode
                episode = scripted_pick_and_place_episode(
                    env, mj_viewer=v, text_label=text_label, cube_id=cube_id
                )
                
                # Save episode to disk
                out_path = os.path.join(out_dir, f"pick_place_episode_{i:03d}.npz")
                save_episode_npz(episode, out_path)
                print(f"[collect] success={episode['success']}")

    if args.replay:
        # Load and replay a saved episode
        npz_path = os.path.abspath(args.replay)
        print(f"[replay] Loading episode from: {npz_path}")
        episode = load_episode_npz(npz_path)
        language_str = episode.get('language', 'N/A')
        print(
            f"[replay] task={episode['task']}, language={language_str}, success={episode['success']}"
        )
        replay_episode(env, episode)


if __name__ == "__main__":
    main()
