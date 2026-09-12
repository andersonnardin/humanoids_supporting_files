"""Single-window front/side pin-drag editor for a stick humanoid."""

from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WINDOW_WIDTH = 1320
WINDOW_HEIGHT = 720
CANVAS_WIDTH = 520
CANVAS_HEIGHT = 680
SCALE = 360.0
FLOOR_Y = 625.0
KEYFRAME_FILE = Path(__file__).resolve().with_name("keyframes.json")

HANDLE_NAMES = (
    "torso",
    "pelvis",
    "left hand",
    "right hand",
    "left elbow",
    "right elbow",
    "left knee",
    "right knee",
    "left foot",
    "right foot",
)
DEFAULT_PINS = ("left foot", "right foot", "left hand")

COLORS = {
    "torso": "#9333ea",
    "pelvis": "#7c3aed",
    "left hand": "#2563eb",
    "right hand": "#d97706",
    "left elbow": "#38bdf8",
    "right elbow": "#f59e0b",
    "left knee": "#14b8a6",
    "right knee": "#10b981",
    "left foot": "#0f766e",
    "right foot": "#047857",
}


@dataclass(frozen=True)
class Link:
    first: str
    second: str
    width: int


LINKS = (
    Link("pelvis", "torso", 9),
    Link("pelvis", "left knee", 7),
    Link("left knee", "left foot", 7),
    Link("pelvis", "right knee", 7),
    Link("right knee", "right foot", 7),
    Link("torso", "left elbow", 6),
    Link("left elbow", "left hand", 6),
    Link("torso", "right elbow", 6),
    Link("right elbow", "right hand", 6),
)


def initial_pose() -> dict[str, np.ndarray]:
    """Return a slightly bent standing pose in x, y, z coordinates."""
    return {
        "pelvis": np.array([0.00, 0.00, 0.82]),
        "torso": np.array([0.00, 0.00, 1.24]),
        "left knee": np.array([-0.12, 0.08, 0.43]),
        "right knee": np.array([0.12, 0.08, 0.43]),
        "left foot": np.array([-0.12, 0.00, 0.04]),
        "right foot": np.array([0.12, 0.00, 0.04]),
        "left elbow": np.array([-0.31, 0.03, 0.98]),
        "right elbow": np.array([0.31, 0.03, 0.98]),
        "left hand": np.array([-0.39, 0.00, 0.71]),
        "right hand": np.array([0.39, 0.00, 0.71]),
    }


def link_lengths(pose: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    return {
        (link.first, link.second): float(
            np.linalg.norm(pose[link.second] - pose[link.first])
        )
        for link in LINKS
    }


class PoseModel:
    """Store the shared pose and preserve rigid links during dragging."""

    def __init__(self) -> None:
        self.pose = initial_pose()
        self.lengths = link_lengths(self.pose)
        self.pins = {name: self.pose[name].copy() for name in DEFAULT_PINS}

    def reset(self) -> None:
        self.pose = initial_pose()
        self.lengths = link_lengths(self.pose)
        self.pins = {name: self.pose[name].copy() for name in DEFAULT_PINS}

    def set_pin(self, name: str, enabled: bool) -> None:
        if enabled:
            self.pins[name] = self.pose[name].copy()
        else:
            self.pins.pop(name, None)

    def maximum_link_error(self) -> float:
        return max(
            abs(
                np.linalg.norm(self.pose[link.second] - self.pose[link.first])
                - self.lengths[(link.first, link.second)]
            )
            for link in LINKS
        )

    def target_around_pinned_neighbor(
        self, name: str, target: np.ndarray, horizontal_axis: int
    ) -> np.ndarray:
        """Let a link rotate around a pinned neighbor without stretching."""
        for link in LINKS:
            if link.first == name:
                neighbor = link.second
            elif link.second == name:
                neighbor = link.first
            else:
                continue
            if neighbor not in self.pins:
                continue

            center = self.pins[neighbor]
            length = self.lengths[(link.first, link.second)]
            hidden_axis = 1 if horizontal_axis == 0 else 0
            hidden_offset = target[hidden_axis] - center[hidden_axis]
            visible_radius = np.sqrt(
                max(length * length - hidden_offset * hidden_offset, 0.0)
            )
            visible_offset = np.array(
                [
                    target[horizontal_axis] - center[horizontal_axis],
                    target[2] - center[2],
                ]
            )
            visible_norm = float(np.linalg.norm(visible_offset))
            if visible_norm < 1e-9:
                current = self.pose[name]
                visible_offset = np.array(
                    [
                        current[horizontal_axis] - center[horizontal_axis],
                        current[2] - center[2],
                    ]
                )
                visible_norm = max(float(np.linalg.norm(visible_offset)), 1e-9)

            projected = target.copy()
            projected[horizontal_axis] = (
                center[horizontal_axis]
                + visible_radius * visible_offset[0] / visible_norm
            )
            projected[2] = center[2] + visible_radius * visible_offset[1] / visible_norm
            return projected
        return target

    def solve_drag(self, name: str, target: np.ndarray) -> None:
        """Move one handle while satisfying pins and immutable link lengths."""
        previous_pose = {key: value.copy() for key, value in self.pose.items()}
        fixed = {key: value.copy() for key, value in self.pins.items()}
        fixed[name] = target.copy()
        self.pose[name] = target.copy()

        for _ in range(250):
            for link in LINKS:
                first = self.pose[link.first]
                second = self.pose[link.second]
                difference = second - first
                distance = float(np.linalg.norm(difference))
                if distance < 1e-9:
                    continue
                correction = difference * (
                    (distance - self.lengths[(link.first, link.second)]) / distance
                )
                first_free = link.first not in fixed
                second_free = link.second not in fixed
                if first_free and second_free:
                    self.pose[link.first] += 0.5 * correction
                    self.pose[link.second] -= 0.5 * correction
                elif first_free:
                    self.pose[link.first] += correction
                elif second_free:
                    self.pose[link.second] -= correction

            for fixed_name, fixed_position in fixed.items():
                self.pose[fixed_name] = fixed_position.copy()
            if self.maximum_link_error() < 1e-6:
                break

        if self.maximum_link_error() > 1e-4:
            self.pose = previous_pose


class ViewWindow:
    """Interactive orthographic view embedded in the application window."""

    def __init__(
        self,
        app: "SingleWindowPinDragApplication",
        host: tk.Misc,
        _title: str,
        horizontal_axis: int,
    ) -> None:
        self.app = app
        self.horizontal_axis = horizontal_axis
        self.dragging: str | None = None
        self.canvas = tk.Canvas(
            host,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#f8fafc",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", lambda _event: self.app.draw_all())

    def origin_x(self) -> float:
        return max(self.canvas.winfo_width(), CANVAS_WIDTH) / 2.0

    def project(self, point: np.ndarray) -> tuple[float, float]:
        horizontal_value = point[self.horizontal_axis]
        if self.horizontal_axis == 0:
            horizontal_value = -horizontal_value
        return (
            self.origin_x() + SCALE * horizontal_value,
            FLOOR_Y - SCALE * point[2],
        )

    def unproject(self, x: float, y: float, current: np.ndarray) -> np.ndarray:
        target = current.copy()
        horizontal_value = (x - self.origin_x()) / SCALE
        if self.horizontal_axis == 0:
            horizontal_value = -horizontal_value
        target[self.horizontal_axis] = horizontal_value
        target[2] = (FLOOR_Y - y) / SCALE
        return target

    def nearest_handle(self, x: float, y: float) -> str | None:
        candidates = []
        for name in HANDLE_NAMES:
            px, py = self.project(self.app.model.pose[name])
            candidates.append((float(np.hypot(x - px, y - py)), name))
        distance, name = min(candidates)
        return name if distance <= 17.0 else None

    def on_press(self, event: tk.Event) -> None:
        name = self.nearest_handle(event.x, event.y)
        if name is not None and name not in self.app.model.pins:
            self.dragging = name

    def on_drag(self, event: tk.Event) -> None:
        if self.dragging is None:
            return
        current = self.app.model.pose[self.dragging]
        target = self.unproject(event.x, event.y, current)
        target = self.app.model.target_around_pinned_neighbor(
            self.dragging, target, self.horizontal_axis
        )
        self.app.model.solve_drag(self.dragging, target)
        self.app.draw_all()

    def on_release(self, _event: tk.Event) -> None:
        self.dragging = None

    def project_link(self, first: str, second: str, width: int) -> None:
        x1, y1 = self.project(self.app.model.pose[first])
        x2, y2 = self.project(self.app.model.pose[second])
        self.canvas.create_line(
            x1, y1, x2, y2, fill="#334155", width=width, capstyle="round"
        )

    def draw(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), CANVAS_WIDTH)
        self.canvas.create_line(
            28, FLOOR_Y, width - 28, FLOOR_Y, fill="#64748b", width=3
        )
        for link in LINKS:
            self.project_link(link.first, link.second, link.width)

        head = self.app.model.pose["torso"] + np.array([0.0, 0.0, 0.12])
        hx, hy = self.project(head)
        self.canvas.create_oval(
            hx - 23,
            hy - 23,
            hx + 23,
            hy + 23,
            fill="#ffffff",
            outline="#334155",
            width=3,
        )

        for name in HANDLE_NAMES:
            x, y = self.project(self.app.model.pose[name])
            pinned = name in self.app.model.pins
            main_handle = name in (
                "torso",
                "pelvis",
                "left foot",
                "right foot",
                "left hand",
                "right hand",
            )
            radius = 10 if main_handle else 7
            if pinned:
                self.canvas.create_rectangle(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill="#dc2626",
                    outline="#7f1d1d",
                    width=2,
                )
            else:
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=COLORS[name],
                    outline="#ffffff",
                    width=2,
                )
            label = name.replace("left ", "L ").replace("right ", "R ")
            self.canvas.create_text(
                x,
                y - 18,
                text=label,
                fill="#7f1d1d" if pinned else "#334155",
                font=("Segoe UI", 8, "bold"),
            )

        axis_name = "left / right" if self.horizontal_axis == 0 else "forward / back"
        self.canvas.create_text(
            18,
            18,
            anchor="nw",
            text=f"Drag a circular handle: {axis_name} and height",
            fill="#475569",
            font=("Segoe UI", 10),
        )


class EmbeddedViewHost(tk.Frame):
    """Frame adapter that lets the existing view render inside one window."""

    def title(self, _text: str) -> None:
        pass

    def geometry(self, _geometry: str) -> None:
        pass

    def minsize(self, _width: int, _height: int) -> None:
        pass


class SingleWindowPinDragApplication:
    """Place synchronized frontal and lateral editors in one Tkinter window."""

    def __init__(self) -> None:
        self.model = PoseModel()
        self.root = tk.Tk()
        self.root.title("Whole-Body Pin/Drag Editor - Front and Side Views")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(1100, 640)
        self.pin_variables: dict[str, tk.BooleanVar] = {}
        self.keyframe_status = tk.StringVar(master=self.root)

        controls = tk.Frame(
            self.root,
            width=230,
            bg="#ffffff",
            padx=14,
            pady=14,
        )
        controls.pack(side="right", fill="y")
        controls.pack_propagate(False)

        views_area = tk.Frame(self.root, bg="#e2e8f0")
        views_area.pack(side="left", fill="both", expand=True)
        views_area.grid_rowconfigure(0, weight=1)
        views_area.grid_columnconfigure(0, weight=1, uniform="views")
        views_area.grid_columnconfigure(1, weight=1, uniform="views")

        front_container = self.make_view_container(views_area, "FRONT VIEW", 0)
        side_container = self.make_view_container(views_area, "SIDE VIEW", 1)
        front_container.grid(row=0, column=0, sticky="nsew", padx=(0, 1))
        side_container.grid(row=0, column=1, sticky="nsew", padx=(1, 0))

        front_host = front_container.view_host
        side_host = side_container.view_host
        self.front = ViewWindow(self, front_host, "Front View", 0)
        self.side = ViewWindow(self, side_host, "Side View", 1)
        self.views = (self.front, self.side)

        self.build_controls(controls)
        self.update_keyframe_status()
        self.root.bind("r", lambda _event: self.reset())
        self.root.after_idle(self.draw_all)

    def make_view_container(
        self, parent: tk.Misc, title: str, column: int
    ) -> tk.Frame:
        container = tk.Frame(parent, bg="#f8fafc")
        tk.Label(
            container,
            text=title,
            bg="#ffffff",
            fg="#0f172a",
            pady=8,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="top", fill="x")
        host = EmbeddedViewHost(container, bg="#f8fafc")
        host.pack(side="top", fill="both", expand=True)
        container.view_host = host
        return container

    def build_controls(self, panel: tk.Frame) -> None:
        tk.Label(
            panel,
            text="PIN / DRAG",
            bg="#ffffff",
            fg="#0f172a",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 5))
        tk.Label(
            panel,
            text=(
                "Both panels edit the same pose. Red squares are pinned; "
                "unpinned circles can be dragged."
            ),
            bg="#ffffff",
            fg="#475569",
            justify="left",
            wraplength=195,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 10))

        for name in HANDLE_NAMES:
            variable = tk.BooleanVar(master=self.root, value=name in DEFAULT_PINS)
            self.pin_variables[name] = variable
            tk.Checkbutton(
                panel,
                text=f"Pin {name}",
                variable=variable,
                command=lambda endpoint=name: self.toggle_pin(endpoint),
                bg="#ffffff",
                activebackground="#ffffff",
                selectcolor="#ffffff",
                fg="#0f172a",
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=1)

        tk.Frame(panel, height=1, bg="#e2e8f0").pack(fill="x", pady=10)
        tk.Button(
            panel,
            text="Save Keyframe",
            command=self.save_keyframe,
            bg="#166534",
            fg="#ffffff",
            activebackground="#15803d",
            activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"),
        ).pack(fill="x", pady=(0, 7))
        tk.Label(
            panel,
            textvariable=self.keyframe_status,
            bg="#ffffff",
            fg="#475569",
            justify="left",
            wraplength=195,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 10))
        tk.Button(
            panel,
            text="Reset pose",
            command=self.reset,
            font=("Segoe UI", 9),
        ).pack(fill="x")

    def read_keyframes(self) -> list[dict[str, object]]:
        if not KEYFRAME_FILE.exists():
            return []
        try:
            data = json.loads(KEYFRAME_FILE.read_text(encoding="utf-8"))
            keyframes = data.get("keyframes", [])
            return keyframes if isinstance(keyframes, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def update_keyframe_status(self) -> None:
        count = len(self.read_keyframes())
        noun = "keyframe" if count == 1 else "keyframes"
        self.keyframe_status.set(f"{count} saved {noun}\n{KEYFRAME_FILE.name}")

    def save_keyframe(self) -> None:
        """Append the current complete 3D posture to the shared motion file."""
        keyframes = self.read_keyframes()
        keyframes.append(
            {
                "index": len(keyframes),
                "pose": {
                    name: [float(value) for value in point]
                    for name, point in self.model.pose.items()
                },
                "pins": sorted(self.model.pins),
                "pin_targets": {
                    name: [float(value) for value in point]
                    for name, point in self.model.pins.items()
                },
            }
        )
        document = {
            "format": "humanoid-motion-stick-keyframes",
            "version": 1,
            "coordinate_system": "x=left/right, y=forward/back, z=up",
            "keyframes": keyframes,
        }
        temporary_file = KEYFRAME_FILE.with_suffix(".tmp")
        temporary_file.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        temporary_file.replace(KEYFRAME_FILE)
        self.update_keyframe_status()

    def toggle_pin(self, name: str) -> None:
        self.model.set_pin(name, self.pin_variables[name].get())
        self.draw_all()

    def reset(self) -> None:
        self.model.reset()
        for name, variable in self.pin_variables.items():
            variable.set(name in DEFAULT_PINS)
        self.draw_all()

    def draw_all(self) -> None:
        for view in self.views:
            view.draw()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    SingleWindowPinDragApplication().run()


if __name__ == "__main__":
    main()
