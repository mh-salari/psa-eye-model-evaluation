function [ellipse, cr_px] = project(s, c, eye, target, diameter_m, shift_m)
%  The pupil ellipse, [center x, center y, x radius, y radius, angle] (px), and the corneal reflection of the setup's
%  illuminator (px, or [] when et_simul finds none) in the camera image, with the eye looking at the target, for a
%  physical pupil diameter and an eye-local shift of the pupil center (m).
%
%  The ellipse is fitted to the convex hull of the refracted pupil boundary, as the measured pupils are fitted.
    e = eye_make(s.cornea_radius_m, s.rest_orientation);
    e.across_pupil = diameter_m / 2 * [1 0 0 0]';
    e.up_pupil = diameter_m / 2 * [0 1 0 0]';
    % The setup's eye position is the corneal apex of the unrotated eye; et_simul's eye origin is its center, behind
    % the apex.
    e.trans(1:3, 4) = eye.position - s.rest_orientation * e.pos_apex(1:3);
    e.pos_pupil = e.pos_pupil + [eye.pupil_offset_m + shift_m; 0; 0];
    e = eye_look_at(e, [target; 1]);
    boundary = eye_get_pupil_image(e, c, s.pupil_boundary_points);
    hull = convhull(boundary(1, :), boundary(2, :));
    hull = hull(1:end - 1);   % convhull repeats the first vertex to close the loop
    ellipse = reshape(fitellipse_hf(boundary(1, hull), boundary(2, hull)), 1, []);
    if nargout > 1
        light = light_make();
        light.pos = [s.illuminator_position; 1];
        cr_3d = eye_find_cr(e, light, c);
        cr_px = [];
        if ~isempty(cr_3d)
            cr_px = camera_project(c, cr_3d)';
        end
    end
end
