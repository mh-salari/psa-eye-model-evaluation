"""The setup in PyEtSimul: camera, illuminator, eye model, eye and target positions, built one way everywhere."""

import math
from dataclasses import replace

import numpy as np
from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.camera import CameraImage
from pyetsimul.core.cornea import ConicCornea
from pyetsimul.core.eye_model import EyeModel
from pyetsimul.core.off_axis_pupil import OffAxisPupilConfig
from pyetsimul.optics.pupil_imaging import fit_ellipse
from pyetsimul.types import Position3D, RotationMatrix

from psa_eye_model_evaluation.experiment_setup import (
    CAMERA_POSITIONS_MM,
    CENTER_TARGET,
    EYE_POSITIONS_MM,
    ILLUMINATOR_POSITION_MM,
    SCREEN_POSITIONS,
)
from psa_eye_model_evaluation.recordings import camera_roll_deg

# The same pupil center as in the annotated images: the center of an ellipse fitted to the convex hull of the boundary.
PUPIL_CENTER_METHOD = "convex_hull"
# The physical pupil diameter search stops once the simulated apparent pupil diameter at the center target is this
# close to the measured one, and fails loudly if it has not got there within the iteration limit.
PUPIL_DIAMETER_TOLERANCE_MM = 1e-4
PUPIL_DIAMETER_MAX_ITERATIONS = 20

EYE_POSITIONS = {eye: Position3D(*position) for eye, position in EYE_POSITIONS_MM.items()}
CAMERA_POSITIONS = {camera: Position3D(*position) for camera, position in CAMERA_POSITIONS_MM.items()}
ILLUMINATOR_POSITION = Position3D(*ILLUMINATOR_POSITION_MM)
# The targets lie on the screen plane, at zero distance.
TARGET_POSITIONS = {target: Position3D(x, height, 0.0) for target, (x, height) in SCREEN_POSITIONS.items()}


def build_camera(camera: str) -> Camera:
    """The camera at its setup position, aimed at the midpoint between the eyes, with the real camera's image axes.

    The frame is passed to point_at explicitly, so that the simulated image has the recording's axes (image x toward
    screen right, image y downward) and the real camera's roll, neither of which a pan-tilt aim sets. The frame puts
    the line between the eyes along image x, and the roll measured in the recordings is then applied about the view
    axis.
    """
    left, right = (np.array(EYE_POSITIONS_MM[eye]) for eye in ("left_eye", "right_eye"))
    aim = (left + right) / 2
    view = aim - np.array(CAMERA_POSITIONS_MM[camera])
    view /= np.linalg.norm(view)
    inter_eye = (right - left) / np.linalg.norm(right - left)
    up = np.cross(inter_eye, view)
    up /= np.linalg.norm(up)
    # up is perpendicular to the view axis, so rotating it about that axis needs only the cosine and sine terms.
    roll = math.radians(camera_roll_deg(camera))
    up = up * math.cos(roll) + np.cross(view, up) * math.sin(roll)
    z = -up
    x = np.cross(view, up)
    x /= np.linalg.norm(x)

    simulated = Camera()
    simulated.position = CAMERA_POSITIONS[camera]
    simulated.point_at(
        Position3D(*aim), RotationMatrix(np.column_stack([x, np.cross(z, x), z]), validate_handedness=False)
    )
    return simulated


def build_illuminator() -> Light:
    """The infrared illuminator at its setup position."""
    return Light(position=ILLUMINATOR_POSITION)


def population_anatomy() -> dict[str, float]:
    """PyEtSimul's population eye, under the keys of build_eye_model's ``anatomy``."""
    model = EyeModel()
    off_axis_pupil = replace(model.off_axis_pupil)
    off_axis_pupil.resolve_defaults()
    return {
        "anterior_radius": float(model.cornea.anterior_radius),
        "alpha_deg": float(model.fovea_alpha_deg),
        "beta_deg": float(model.fovea_beta_deg),
        "off_axis_nasal_mm": float(off_axis_pupil.nasal_mm),
        "off_axis_superior_mm": float(off_axis_pupil.superior_mm),
    }


def build_eye_model(anatomy: dict[str, float] | None = None) -> EyeModel:
    """PyEtSimul's population eye, or that eye with the anatomy of a fit_eye_parameters.py run in its place.

    ``anatomy`` holds the corneal apex radius, the horizontal and vertical angle kappa and the nasal and superior
    off-axis pupil, under the keys fit_eye_parameters.py writes.
    """
    if anatomy is None:
        return EyeModel()
    cornea = ConicCornea(use_posterior_surface=True)
    cornea.anterior_radius = anatomy["anterior_radius"]
    return EyeModel(
        cornea=cornea,
        fovea_alpha_deg=anatomy["alpha_deg"],
        fovea_beta_deg=anatomy["beta_deg"],
        off_axis_pupil=OffAxisPupilConfig(
            enabled=True, nasal_mm=anatomy["off_axis_nasal_mm"], superior_mm=anatomy["off_axis_superior_mm"]
        ),
    )


def build_eye(model: EyeModel, eye: str) -> Eye:
    """A PyEtSimul eye built from the model, at the eye's setup position."""
    simulated = Eye(model=model, which_eye=eye.removesuffix("_eye"))
    simulated.position = EYE_POSITIONS[eye]
    return simulated


def mm_per_pixel(camera: Camera, eye: str) -> float:
    """Millimeters per image pixel at the eye: its distance from the camera over the focal length in pixels."""
    position = camera.position
    distance = np.linalg.norm(np.array(EYE_POSITIONS_MM[eye]) - np.array([position.x, position.y, position.z]))
    return float(distance / camera.camera_matrix.focal_length)


def apparent_pupil_diameter_mm(image: CameraImage, camera: Camera, eye: str) -> float:
    """The pupil diameter in a simulated image, as in the recordings: the mean of the fitted ellipse's two axes."""
    boundary = np.array([[point.x, point.y] for point in image.pupil_boundary])
    fitted = fit_ellipse(boundary)
    if fitted is None:
        raise ValueError(f"no ellipse fits the simulated pupil boundary of the {eye}")
    _center_x, _center_y, major_axis, minor_axis, _angle = fitted
    return mm_per_pixel(camera, eye) * (major_axis + minor_axis) / 2


def physical_pupil_diameter_mm(model: EyeModel, camera: Camera, eye: str, center_target_mm: float) -> float:
    """The physical pupil diameter whose apparent diameter at the center target is ``center_target_mm``.

    Each step rescales the physical diameter by the ratio still missing, which converges because the apparent diameter
    is close to proportional to the physical one.
    """
    simulated = build_eye(model, eye)
    physical_mm = center_target_mm
    for _ in range(PUPIL_DIAMETER_MAX_ITERATIONS):
        simulated.set_pupil_diameter(physical_mm)
        simulated.look_at(TARGET_POSITIONS[CENTER_TARGET])
        apparent_mm = apparent_pupil_diameter_mm(camera.take_image(simulated), camera, eye)
        if abs(apparent_mm - center_target_mm) <= PUPIL_DIAMETER_TOLERANCE_MM:
            return physical_mm
        physical_mm *= center_target_mm / apparent_mm
    raise ValueError(
        f"the {eye}'s physical pupil diameter for an apparent {center_target_mm:.3f} mm at the center target did not "
        f"converge within {PUPIL_DIAMETER_MAX_ITERATIONS} iterations"
    )
