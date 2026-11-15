function [ellipse, glint_px] = project(sg, pose, stop_radius_mm, stop_shift_mm, perimeter_points)
%  The ellipse fitted to the entrance pupil, [center x, center y, semi-major axis, semi-minor axis, tilt] (px), and the
%  glint (px) in the image, for a physical stop radius and a shift of the stop center along the eye's horizontal and
%  vertical axes (mm).
    sg.eye.stop.center(2:3) = sg.eye.stop.center(2:3) + stop_shift_mm;
    pose(4) = stop_radius_mm;
    [transparent, glint_px] = projectModelEye(pose, sg, 'nStopPerimPoints', perimeter_points);
    ellipse = ellipse_transparent2ex(transparent)';
end
