function [center_m, diameter_m] = apparent_pupil(s, c, eye, target, physical_diameter_m, shift_m)
%  The apparent pupil center (m at the eye, image axes) and apparent pupil diameter (m) at one target, for a physical
%  pupil diameter and an eye-local shift of the pupil center.
    ellipse = et_simul_scene.project(s, c, eye, target, physical_diameter_m, shift_m);
    scale = et_simul_scene.m_per_px(c, eye);
    center_m = scale * ellipse(1:2);
    % The ellipse's radii are its semi-axes, so the mean of its two axes is their sum.
    diameter_m = scale * sum(ellipse(3:4));
end
