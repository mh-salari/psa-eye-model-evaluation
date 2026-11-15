function [center_mm, diameter_mm] = apparent_pupil(s, sg, pose, stop_radius_mm, stop_shift_mm)
%  The apparent pupil center (mm at the eye, image axes) and apparent pupil diameter (mm) in one pose, for a physical
%  stop radius and a shift of the stop center.
    ellipse = gkamodeleye_scene.project(sg, pose, stop_radius_mm, stop_shift_mm, s.stop_perimeter_points);
    scale = gkamodeleye_scene.mm_per_px(sg);
    center_mm = scale * ellipse(1:2);
    % The ellipse comes as its semi-axes, so the mean of its two axes is their sum.
    diameter_mm = scale * sum(ellipse(3:4));
end
