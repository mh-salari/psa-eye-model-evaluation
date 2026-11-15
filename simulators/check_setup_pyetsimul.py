"""A 3D drawing of the simulated setup, to check that PyEtSimul's scene matches the real one.

    uv run python simulators/check_setup_pyetsimul.py

Nothing here is analysis. It places the eyes, both cameras, the illuminator, the screen and the targets at their setup
positions and hands them to PyEtSimul's plot_setup. It opens an interactive window (drag to rotate) and writes
output/check_setup_pyetsimul.png.

The analysis scripts put height on y and distance on z, but matplotlib draws its third axis upright, so plotting those
coordinates directly would stand the eye-to-screen distance on end. Every position is therefore rotated by
(x, y, z) -> (x, -z, y) before plotting: a rotation, so nothing is mirrored and no distance changes.
"""

from itertools import starmap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import EyeModel
from pyetsimul.types import Point2D, Position3D, ScreenGeometry
from pyetsimul.visualization import plot_setup
from pyetsimul.visualization.coordinate_utils import prepare_eye_data_for_plots

from psa_eye_model_evaluation.experiment_setup import (
    CAMERA_LABEL,
    CAMERA_POSITIONS_MM,
    CAMERAS,
    EYE_POSITIONS_MM,
    ILLUMINATOR_POSITION_MM,
    SCREEN_POSITIONS,
    SCREEN_SIZE_MM,
)
from psa_eye_model_evaluation.pyetsimul_scene import CAMERA_POSITIONS, EYE_POSITIONS, ILLUMINATOR_POSITION

OUT = Path(__file__).resolve().parent / "output" / "check_setup_pyetsimul.png"
SCREEN_CENTER = Position3D(0.0, 0.0, 0.0)


def upright(position: Position3D) -> Position3D:
    """The same point with height upright: a rotation about x, so nothing is mirrored or rescaled."""
    return Position3D(position.x, -position.z, position.y)


def print_distances() -> None:
    """The distances that define the setup, to compare against the real one."""
    for eye, position in EYE_POSITIONS_MM.items():
        x, height, distance = position
        print(f"{eye:<10} {distance:.0f} mm from the screen, {height:.0f} mm above its center, x {x:+.0f} mm")
        for camera in CAMERAS:
            to_camera = np.linalg.norm(np.array(position) - np.array(CAMERA_POSITIONS_MM[camera]))
            print(f"           {to_camera:.0f} mm from the {CAMERA_LABEL[camera].lower()}")
    x, height, distance = ILLUMINATOR_POSITION_MM
    print(f"illuminator {distance:.0f} mm from the screen, {height:+.1f} mm from its center height, x {x:+.0f} mm")


def main() -> None:
    """Draw the setup, print its distances, save the drawing and open it."""
    screen_center = upright(SCREEN_CENTER)
    eyes = []
    for eye_name, position in EYE_POSITIONS.items():
        eye = Eye(model=EyeModel(), which_eye=eye_name.removesuffix("_eye"))
        eye.position = upright(position)
        # Rested facing the screen with the drawing's upright direction as its own.
        eye.set_rest_orientation_at_target(screen_center, up=Position3D(0, 0, 1))
        eyes.append(eye)
    cameras = []
    for camera in CAMERAS:
        # Aimed at the midpoint between the eyes; the drawing shows the camera's position and aim, not its roll.
        simulated = Camera()
        simulated.position = upright(CAMERA_POSITIONS[camera])
        simulated.point_at_binocular(eyes[0].position, eyes[1].position)
        cameras.append(simulated)
    lights = [Light(position=upright(ILLUMINATOR_POSITION))]

    figure = plt.figure(figsize=(9.0, 9.0))
    axis = figure.add_subplot(projection="3d")
    targets = [screen_center] * len(eyes)
    prepared = prepare_eye_data_for_plots(eyes, targets, lights, cameras)
    plot_setup(
        axis,
        prepared["eyes_data"],
        targets,
        lights,
        cameras,
        prepared["cr_3d_lists"],
        calib_points=list(starmap(Point2D, SCREEN_POSITIONS.values())),
        screen=ScreenGeometry(width=SCREEN_SIZE_MM[0], height=SCREEN_SIZE_MM[1], plane="xz"),
    )
    axis.set_xlabel("horizontal (mm)")
    axis.set_ylabel("distance from the screen (mm)")
    axis.set_zlabel("height above the screen center (mm)")

    print_distances()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUT}")
    plt.show()


if __name__ == "__main__":
    main()
