#!/usr/bin/env python3
"""PROTOTYPE: edit soft trajectory keyframes in MuJoCo and optimize them.

This uses the UR5e model, tabletop scene, Mink IK, and passive-viewer pattern
from the adjacent SIRL research project. It is intentionally not production
MoveIt integration.

Run on macOS with SIRL's environment:

    /Users/ingui/Documents/Projects/sirl/venv/bin/mjpython \
        interactive_mujoco_trajectory_optimizer_prototype.py

Viewer controls:
    Double-click an end-effector ball to select its trajectory frame.
    Ctrl + right-drag that ball to move it; releasing runs IK and stores an edit.
    Double-click a robot link to select the joint attaching it to its parent.
    Up / Down         increase / decrease that joint by 0.05 rad
    PageUp / PageDown increase / decrease that joint by 0.25 rad
    Left / Right      select the previous / next trajectory frame
    Space             play / pause
    Enter             explicitly store the current frame as a soft keyframe
    O                 optimize and project back to a contact-free path
    C                 clear edits (keep the current candidate)
    R                 reset to the original trajectory
    [ / ]             decrease / increase edit weight
    - / =             decrease / increase smoothness weight
    , / .             narrow / widen each edit's influence
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_SIRL_ROOT = Path("/Users/ingui/Documents/Projects/sirl")


@dataclass
class Edit:
    frame: int
    qpos: np.ndarray


@dataclass
class OptimizerSettings:
    edit_weight: float = 40.0
    smoothness_weight: float = 3.0
    original_weight: float = 0.25
    influence_frames: float = 2.0
    collision_substeps: int = 3


def add_sirl_to_path(sirl_root: Path) -> None:
    root = str(sirl_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


class MuJoCoTrajectoryEditor:
    def __init__(
        self,
        sirl_root: Path,
        trajectory_index: Optional[int],
        settings: OptimizerSettings,
    ) -> None:
        add_sirl_to_path(sirl_root)

        import mujoco
        import yaml
        from sirl.envs.mujoco_robot import MuJoCoRobotEnv

        self.mujoco = mujoco
        self.sirl_root = sirl_root.resolve()
        self.settings = settings
        self.command_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.edits: dict[int, Edit] = {}
        self.frame = 0
        self.playing = False
        self.last_play_step = time.monotonic()
        self.last_message = "Loaded original contact-free trajectory"
        self.viewer = None
        self._perturb_was_active = False
        self._last_perturb_target: Optional[np.ndarray] = None
        self._last_selected_body = 0
        self.selected_joint_address: Optional[int] = None
        self.selected_joint_name: Optional[str] = None
        self._overlay_dirty = True

        config_path = self.sirl_root / "configs/mujoco_robot.yaml"
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        xml_path = self.sirl_root / config["xml_path"]
        self.env = MuJoCoRobotEnv(
            xml_path=xml_path,
            tracked_bodies=config.get("tracked_bodies"),
            excluded_bodies=config.get("excluded_bodies"),
            use_qpos=True,
            use_vel=False,
            traj_horizon=int(config.get("traj_horizon", 21)),
            tabletop=bool(config.get("tabletop", True)),
            ik_max_iters=int(config.get("ik_max_iters", 50)),
            ik_dt=float(config.get("ik_dt", 0.05)),
            ik_damping=float(config.get("ik_damping", 1.0)),
            ik_tol=float(config.get("ik_tol", 1e-3)),
            ik_solver=str(config.get("ik_solver", "daqp")),
            ik_posture_cost=float(config.get("ik_posture_cost", 0.05)),
        )
        self.ee_body_name = mujoco.mj_id2name(
            self.env._model, mujoco.mjtObj.mjOBJ_BODY, self.env._ee_body_id
        )
        frame_count = int(config.get("traj_horizon", 21))
        self.model, self.data = self._build_viewer_model(
            xml_path, bool(config.get("tabletop", True)), frame_count
        )
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name
        )
        self.handle_body_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"trajectory_handle_{frame:02d}"
                )
                for frame in range(frame_count)
            ],
            dtype=int,
        )
        self.handle_geom_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"trajectory_handle_geom_{frame:02d}"
                )
                for frame in range(frame_count)
            ],
            dtype=int,
        )
        self.handle_body_to_frame = {
            int(body_id): frame for frame, body_id in enumerate(self.handle_body_ids)
        }
        self.joint_ranges = self._joint_ranges()

        self.original, self.pool_index = self._load_original_trajectory(trajectory_index)
        self.candidate = self.original.copy()
        self._set_frame_qpos()
        self._print_state()

    def _build_viewer_model(self, xml_path: Path, tabletop: bool, frame_count: int):
        """Build SIRL's scene plus selectable, non-colliding mocap frame handles."""
        spec = self.mujoco.MjSpec.from_file(str(xml_path))
        if tabletop:
            from sirl.envs.scene_builder import add_tabletop_objects

            add_tabletop_objects(spec)

        for frame in range(frame_count):
            body = spec.worldbody.add_body()
            body.name = f"trajectory_handle_{frame:02d}"
            body.mocap = True
            body.pos = [0.0, 0.0, -5.0]
            geom = body.add_geom()
            geom.name = f"trajectory_handle_geom_{frame:02d}"
            geom.type = self.mujoco.mjtGeom.mjGEOM_SPHERE
            geom.size = [0.024, 0.024, 0.024]
            geom.rgba = [0.05, 0.75, 1.0, 0.72]
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 1

        model = spec.compile()
        data = self.mujoco.MjData(model)
        if model.nq != self.env._model.nq:
            raise RuntimeError("Selectable handle bodies unexpectedly changed the robot qpos layout")
        return model, data

    def _joint_ranges(self) -> np.ndarray:
        ranges = np.zeros((self.model.nq, 2), dtype=float)
        for joint_id in range(self.model.njnt):
            address = self.model.jnt_qposadr[joint_id]
            if address >= self.model.nq:
                continue
            if self.model.jnt_limited[joint_id]:
                ranges[address] = self.model.jnt_range[joint_id]
            else:
                ranges[address] = [-2.0 * np.pi, 2.0 * np.pi]
        return ranges

    def _load_original_trajectory(self, requested_index: Optional[int]) -> tuple[np.ndarray, int]:
        pool_path = self.sirl_root / "data/mujoco_pool_jointspace_qpos.npy"
        collision_path = self.sirl_root / "data/mujoco_pool_jointspace_collision.npy"
        pool = np.load(pool_path, mmap_mode="r")
        flagged = np.load(collision_path, mmap_mode="r")

        if requested_index is not None:
            indices = [requested_index]
        else:
            indices = np.flatnonzero(~flagged)[:500]

        for index in indices:
            trajectory = np.asarray(pool[index], dtype=float).copy()
            if not self.trajectory_has_contact(trajectory, substeps=self.settings.collision_substeps):
                return trajectory, int(index)

        if requested_index is not None:
            raise ValueError(
                f"Trajectory {requested_index} is in contact in the current MuJoCo scene; "
                "omit --trajectory-index to choose a valid one automatically."
            )
        raise RuntimeError("Could not find a contact-free trajectory in the first 500 valid pool rows")

    def _set_frame_qpos(self) -> None:
        self.data.qpos[: self.model.nq] = self.candidate[self.frame]
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def qpos_has_contact(self, qpos: np.ndarray) -> bool:
        self.data.qpos[: self.model.nq] = qpos
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return self.data.ncon > 0

    def trajectory_has_contact(self, trajectory: np.ndarray, substeps: int = 3) -> bool:
        for start, end in zip(trajectory[:-1], trajectory[1:]):
            for alpha in np.linspace(0.0, 1.0, substeps, endpoint=False):
                if self.qpos_has_contact((1.0 - alpha) * start + alpha * end):
                    return True
        return self.qpos_has_contact(trajectory[-1])

    def _edited_reference(self) -> tuple[np.ndarray, np.ndarray]:
        reference = self.original.copy()
        influence = np.zeros(len(self.original), dtype=float)
        displacement = np.zeros_like(self.original)
        frames = np.arange(len(self.original), dtype=float)
        sigma = max(0.25, self.settings.influence_frames)

        for edit in self.edits.values():
            alpha = np.exp(-0.5 * ((frames - edit.frame) / sigma) ** 2)
            delta = wrap_angle_delta(edit.qpos - self.original[edit.frame])
            displacement += alpha[:, None] * delta
            influence += alpha

        reference += displacement
        reference[0] = self.original[0]
        reference[-1] = self.original[-1]
        return reference, influence

    def optimize(self) -> None:
        if not self.edits:
            self.last_message = "Nothing to optimize: drag and store at least one frame"
            self._overlay_dirty = True
            return

        reference, influence = self._edited_reference()
        frame_count = len(self.original)
        second_difference = np.zeros((frame_count - 2, frame_count), dtype=float)
        for row in range(frame_count - 2):
            second_difference[row, row : row + 3] = [1.0, -2.0, 1.0]

        edit_matrix = np.diag(influence)
        hessian = (
            self.settings.edit_weight * edit_matrix
            + self.settings.original_weight * np.eye(frame_count)
            + self.settings.smoothness_weight * (second_difference.T @ second_difference)
        )
        rhs = (
            self.settings.edit_weight * (edit_matrix @ reference)
            + self.settings.original_weight * self.original
        )

        # The actual start and the original goal are hard constraints.
        endpoint_weight = 1e8
        for endpoint in (0, frame_count - 1):
            hessian[endpoint, endpoint] += endpoint_weight
            rhs[endpoint] += endpoint_weight * self.original[endpoint]

        unconstrained = np.linalg.solve(hessian, rhs)
        unconstrained = np.clip(
            unconstrained,
            self.joint_ranges[:, 0][None, :],
            self.joint_ranges[:, 1][None, :],
        )

        alpha, projected = self._project_to_contact_free(unconstrained)
        self.candidate = projected
        self._set_frame_qpos()
        self.last_message = (
            f"Optimized; retained {100.0 * alpha:.1f}% of the unconstrained deformation"
        )
        self._overlay_dirty = True
        self._clear_handle_selection()
        self._print_state()

    def _project_to_contact_free(self, target: np.ndarray) -> tuple[float, np.ndarray]:
        if not self.trajectory_has_contact(target, self.settings.collision_substeps):
            return 1.0, target

        # The original was selected as contact-free. Search the straight line in
        # trajectory space for the largest retained deformation that is feasible.
        safe_alpha = 0.0
        unsafe_alpha = 1.0
        for alpha in np.linspace(0.975, 0.0, 40):
            trial = self.original + alpha * (target - self.original)
            if not self.trajectory_has_contact(trial, self.settings.collision_substeps):
                safe_alpha = float(alpha)
                break
            unsafe_alpha = float(alpha)

        for _ in range(10):
            alpha = 0.5 * (safe_alpha + unsafe_alpha)
            trial = self.original + alpha * (target - self.original)
            if self.trajectory_has_contact(trial, self.settings.collision_substeps):
                unsafe_alpha = alpha
            else:
                safe_alpha = alpha

        projected = self.original + safe_alpha * (target - self.original)
        return safe_alpha, projected

    def store_edit(self) -> None:
        if self.frame in (0, len(self.candidate) - 1):
            self.last_message = "Start and goal are fixed; choose an interior frame"
            self._overlay_dirty = True
            return
        self.edits[self.frame] = Edit(self.frame, self.candidate[self.frame].copy())
        self.last_message = f"Stored soft keyframe at frame {self.frame}"
        self._overlay_dirty = True
        self._print_state()

    def clear_edits(self) -> None:
        self.edits.clear()
        self.last_message = "Cleared edits; retained current candidate"
        self._overlay_dirty = True
        self._print_state()

    def reset(self) -> None:
        self.candidate = self.original.copy()
        self.edits.clear()
        self.frame = 0
        self.playing = False
        self.last_message = "Reset to original trajectory"
        self._set_frame_qpos()
        self._overlay_dirty = True
        self._clear_handle_selection()
        self._print_state()

    def _edit_rmse(self) -> float:
        if not self.edits:
            return 0.0
        error = np.stack(
            [wrap_angle_delta(self.candidate[i] - edit.qpos) for i, edit in self.edits.items()]
        )
        return float(np.sqrt(np.mean(np.square(error))))

    def _roughness(self) -> float:
        acceleration = self.candidate[:-2] - 2.0 * self.candidate[1:-1] + self.candidate[2:]
        return float(np.mean(np.square(acceleration)))

    def _print_state(self) -> None:
        edits = ",".join(str(frame) for frame in sorted(self.edits)) or "none"
        collision = self.trajectory_has_contact(
            self.candidate, substeps=self.settings.collision_substeps
        )
        print(
            f"[{self.last_message}] frame={self.frame}/{len(self.candidate) - 1} "
            f"edits={edits} contact={'YES' if collision else 'no'} "
            f"edit_rmse={self._edit_rmse():.4f}rad roughness={self._roughness():.6f} "
            f"weights(edit={self.settings.edit_weight:.2f}, "
            f"smooth={self.settings.smoothness_weight:.2f}, "
            f"original={self.settings.original_weight:.2f}, "
            f"width={self.settings.influence_frames:.2f} frames)"
        )
        self._set_frame_qpos()

    def _end_effector_positions(self, trajectory: np.ndarray) -> np.ndarray:
        positions = []
        for qpos in trajectory:
            self.data.qpos[: self.model.nq] = qpos
            self.mujoco.mj_forward(self.model, self.data)
            positions.append(self.data.xpos[self.ee_body_id].copy())
        self._set_frame_qpos()
        return np.asarray(positions)

    def _draw_path(self, scene, positions: np.ndarray, rgba: np.ndarray, radius: float) -> None:
        for start, end in zip(positions[:-1], positions[1:]):
            if scene.ngeom >= scene.maxgeom:
                return
            self.mujoco.mjv_connector(
                scene.geoms[scene.ngeom],
                self.mujoco.mjtGeom.mjGEOM_CAPSULE,
                radius,
                start.astype(float),
                end.astype(float),
            )
            scene.geoms[scene.ngeom].rgba[:] = rgba
            scene.ngeom += 1

    def _update_handle_balls(self, positions: np.ndarray) -> None:
        """Move and recolor the selectable mocap balls for all trajectory frames."""
        for frame, (body_id, geom_id, position) in enumerate(
            zip(self.handle_body_ids, self.handle_geom_ids, positions)
        ):
            mocap_id = self.model.body_mocapid[body_id]
            self.data.mocap_pos[mocap_id] = position
            self.data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
            if frame in self.edits:
                rgba = [1.0, 0.10, 0.80, 0.95]
            elif frame == self.frame:
                rgba = [1.0, 0.55, 0.05, 0.95]
            else:
                rgba = [0.05, 0.75, 1.0, 0.72]
            self.model.geom_rgba[geom_id] = rgba
        self.mujoco.mj_forward(self.model, self.data)

    def _draw_overlays(self) -> None:
        if self.viewer is None or self.viewer.user_scn is None:
            return
        scene = self.viewer.user_scn
        scene.ngeom = 0
        original_positions = self._end_effector_positions(self.original)
        candidate_positions = self._end_effector_positions(self.candidate)
        self._draw_path(scene, original_positions, np.array([0.55, 0.55, 0.55, 0.35]), 0.005)
        self._draw_path(scene, candidate_positions, np.array([0.05, 0.75, 1.00, 0.85]), 0.008)
        self._set_frame_qpos()
        self._update_handle_balls(candidate_positions)

    def _update_text(self) -> None:
        if self.viewer is None:
            return
        edits = ", ".join(str(frame) for frame in sorted(self.edits)) or "none"
        if self.selected_joint_address is None:
            selected_joint = "none (double-click a robot link)"
        else:
            angle = self.candidate[self.frame, self.selected_joint_address]
            selected_joint = f"{self.selected_joint_name} = {angle:+.3f} rad"
        state = (
            f"{self.last_message}\n"
            f"Frame {self.frame}/{len(self.candidate) - 1}   edits: {edits}\n"
            f"selected parent joint: {selected_joint}\n"
            f"edit RMSE {self._edit_rmse():.4f} rad   roughness {self._roughness():.6f}\n"
            f"edit weight {self.settings.edit_weight:.1f}   "
            f"smooth {self.settings.smoothness_weight:.1f}   "
            f"width {self.settings.influence_frames:.1f} frames"
        )
        controls = (
            "DOUBLE-CLICK ball -> frame; CTRL+right-drag ball -> IK edit\n"
            "DOUBLE-CLICK robot link -> parent joint; UP/DOWN +/-0.05; PGUP/PGDN +/-0.25\n"
            "LEFT/RIGHT frame   SPACE play   ENTER store   O optimize   C clear   R reset\n"
            "[/] edit weight   -/= smoothness   ,/. influence width"
        )
        self.viewer.set_texts(
            [
                (
                    self.mujoco.mjtFontScale.mjFONTSCALE_150,
                    self.mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "SOFT-KEYFRAME TRAJECTORY OPTIMIZER — PROTOTYPE",
                    state,
                ),
                (
                    self.mujoco.mjtFontScale.mjFONTSCALE_100,
                    self.mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    "CONTROLS",
                    controls,
                ),
            ]
        )

    def _clear_handle_selection(self) -> None:
        if self.viewer is None:
            return
        perturb = self.viewer.perturb
        if int(perturb.select) in self.handle_body_to_frame:
            perturb.active = 0
            perturb.active2 = 0
            perturb.select = 0
        self._last_perturb_target = None

    def _select_parent_joint(self, body_id: int) -> None:
        if body_id <= 0 or body_id >= self.model.nbody:
            return
        if self.model.body_jntnum[body_id] <= 0:
            return
        joint_id = int(self.model.body_jntadr[body_id])
        address = int(self.model.jnt_qposadr[joint_id])
        if address >= self.model.nq:
            return
        self.selected_joint_address = address
        self.selected_joint_name = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        body_name = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        self.last_message = f"Selected {self.selected_joint_name}, parent joint of {body_name}"
        self._overlay_dirty = True

    def _handle_selection_and_perturbation(self) -> None:
        if self.viewer is None:
            return
        perturb = self.viewer.perturb
        selected_body = int(perturb.select)
        active = (int(perturb.active) | int(perturb.active2)) != 0

        if selected_body != self._last_selected_body:
            if selected_body in self.handle_body_to_frame:
                self.frame = self.handle_body_to_frame[selected_body]
                self.playing = False
                self.selected_joint_address = None
                self.selected_joint_name = None
                self.last_message = f"Selected end-effector handle for frame {self.frame}"
                self._set_frame_qpos()
                self._overlay_dirty = True
            elif selected_body > 0:
                self._select_parent_joint(selected_body)
            self._last_selected_body = selected_body

        if active and selected_body in self.handle_body_to_frame:
            selected_frame = self.handle_body_to_frame[selected_body]
            if selected_frame != self.frame:
                self.frame = selected_frame
                self._set_frame_qpos()
            target = perturb.refpos.copy()
            if self._last_perturb_target is None or np.linalg.norm(target - self._last_perturb_target) > 1e-5:
                self.playing = False
                current_quat_wxyz = self.data.xquat[self.ee_body_id].copy()
                current_quat_xyzw = np.array(
                    [
                        current_quat_wxyz[1],
                        current_quat_wxyz[2],
                        current_quat_wxyz[3],
                        current_quat_wxyz[0],
                    ]
                )
                qpos, converged = self.env._solve_ik(
                    target,
                    current_quat_xyzw,
                    self.candidate[self.frame],
                    q_posture=self.candidate[self.frame],
                )
                self.candidate[self.frame] = qpos
                self.last_message = (
                    f"Dragging frame {self.frame} ({'IK converged' if converged else 'IK best effort'})"
                )
                self._last_perturb_target = target
                self._set_frame_qpos()
                self._overlay_dirty = True
        elif self._perturb_was_active and selected_body in self.handle_body_to_frame:
            self.store_edit()
        self._perturb_was_active = active

    def _adjust_selected_joint(self, delta: float) -> None:
        if self.selected_joint_address is None:
            self.last_message = "Double-click a robot link before adjusting a joint"
            self._overlay_dirty = True
            return
        if self.frame in (0, len(self.candidate) - 1):
            self.last_message = "Start and goal are fixed; choose an interior frame"
            self._overlay_dirty = True
            return

        address = self.selected_joint_address
        old_angle = self.candidate[self.frame, address]
        new_angle = float(
            np.clip(
                old_angle + delta,
                self.joint_ranges[address, 0],
                self.joint_ranges[address, 1],
            )
        )
        self.candidate[self.frame, address] = new_angle
        self.edits[self.frame] = Edit(self.frame, self.candidate[self.frame].copy())
        self.last_message = (
            f"{self.selected_joint_name}: {old_angle:+.3f} -> {new_angle:+.3f} rad "
            f"at frame {self.frame}"
        )
        self._set_frame_qpos()
        self._overlay_dirty = True
        self._print_state()

    def enqueue_key(self, keycode: int) -> None:
        self.command_queue.put(str(keycode))

    def _process_commands(self) -> None:
        import glfw

        while True:
            try:
                key = int(self.command_queue.get_nowait())
            except queue.Empty:
                break

            frame_changed = False
            if key == glfw.KEY_LEFT:
                self.frame = max(0, self.frame - 1)
                frame_changed = True
            elif key == glfw.KEY_RIGHT:
                self.frame = min(len(self.candidate) - 1, self.frame + 1)
                frame_changed = True
            elif key == glfw.KEY_SPACE:
                self.playing = not self.playing
                self.last_message = "Playing" if self.playing else "Paused"
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                self.store_edit()
            elif key == glfw.KEY_O:
                self.optimize()
            elif key == glfw.KEY_C:
                self.clear_edits()
            elif key == glfw.KEY_R:
                self.reset()
                frame_changed = True
            elif key == glfw.KEY_UP:
                self._adjust_selected_joint(+0.05)
            elif key == glfw.KEY_DOWN:
                self._adjust_selected_joint(-0.05)
            elif key == glfw.KEY_PAGE_UP:
                self._adjust_selected_joint(+0.25)
            elif key == glfw.KEY_PAGE_DOWN:
                self._adjust_selected_joint(-0.25)
            elif key == glfw.KEY_LEFT_BRACKET:
                self.settings.edit_weight = max(0.1, self.settings.edit_weight / 1.5)
                self.last_message = "Decreased edit weight"
            elif key == glfw.KEY_RIGHT_BRACKET:
                self.settings.edit_weight *= 1.5
                self.last_message = "Increased edit weight"
            elif key == glfw.KEY_MINUS:
                self.settings.smoothness_weight = max(0.01, self.settings.smoothness_weight / 1.5)
                self.last_message = "Decreased smoothness weight"
            elif key == glfw.KEY_EQUAL:
                self.settings.smoothness_weight *= 1.5
                self.last_message = "Increased smoothness weight"
            elif key == glfw.KEY_COMMA:
                self.settings.influence_frames = max(0.25, self.settings.influence_frames - 0.5)
                self.last_message = "Narrowed edit influence"
            elif key == glfw.KEY_PERIOD:
                self.settings.influence_frames += 0.5
                self.last_message = "Widened edit influence"

            if frame_changed:
                self.playing = False
                self._set_frame_qpos()
                self._clear_handle_selection()
            self._overlay_dirty = True

    def _advance_playback(self) -> None:
        if not self.playing or time.monotonic() - self.last_play_step < 0.10:
            return
        self.frame = (self.frame + 1) % len(self.candidate)
        self.last_play_step = time.monotonic()
        self._set_frame_qpos()
        self._clear_handle_selection()
        self._overlay_dirty = True

    def run_viewer(self) -> None:
        import mujoco.viewer as mjviewer

        with mjviewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.enqueue_key,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            self.viewer = viewer
            viewer.cam.lookat[:] = [0.30, 0.0, 0.25]
            viewer.cam.distance = 2.1
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.sync()
            self._clear_handle_selection()
            self._draw_overlays()
            self._update_text()
            viewer.sync()

            while viewer.is_running():
                self._process_commands()
                self._advance_playback()
                self._handle_selection_and_perturbation()
                if self._overlay_dirty:
                    self._draw_overlays()
                    self._update_text()
                    self._overlay_dirty = False
                viewer.sync()
                time.sleep(0.015)

    def run_headless_demo(self, output: Path) -> None:
        middle = len(self.candidate) // 2
        self.frame = middle
        self._set_frame_qpos()
        current_pos = self.data.xpos[self.ee_body_id].copy()
        current_quat_wxyz = self.data.xquat[self.ee_body_id].copy()
        current_quat_xyzw = np.array(
            [current_quat_wxyz[1], current_quat_wxyz[2], current_quat_wxyz[3], current_quat_wxyz[0]]
        )
        target_pos = current_pos + np.array([0.0, -0.16, 0.18])
        qpos, _ = self.env._solve_ik(
            target_pos,
            current_quat_xyzw,
            self.candidate[middle],
            q_posture=self.candidate[middle],
        )
        self.candidate[middle] = qpos
        self.store_edit()
        self.optimize()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            original=self.original,
            optimized=self.candidate,
            edited_frames=np.array(sorted(self.edits)),
            pool_index=np.array(self.pool_index),
        )
        print(f"Saved headless prototype result to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sirl-root", type=Path, default=DEFAULT_SIRL_ROOT)
    parser.add_argument("--trajectory-index", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/mujoco_trajectory_optimizer_prototype.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = MuJoCoTrajectoryEditor(
        sirl_root=args.sirl_root,
        trajectory_index=args.trajectory_index,
        settings=OptimizerSettings(),
    )
    if args.headless:
        app.run_headless_demo(args.output)
    else:
        app.run_viewer()


if __name__ == "__main__":
    main()
