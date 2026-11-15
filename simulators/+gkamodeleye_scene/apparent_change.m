function [shift_mm, diameter_change_mm] = apparent_change(s, sg, pose, dilated_radius_mm, constricted_radius_mm, ...
        stop_shift_mm)
%  Dilated minus constricted apparent pupil center (mm at the eye, screen axes) and apparent pupil diameter (mm) in one
%  pose, the dilated stop moved by stop_shift_mm.
    [dilated_center, dilated_diameter] = gkamodeleye_scene.apparent_pupil(s, sg, pose, dilated_radius_mm, ...
        stop_shift_mm);
    [constricted_center, constricted_diameter] = gkamodeleye_scene.apparent_pupil(s, sg, pose, ...
        constricted_radius_mm, [0, 0]);
    shift_mm = s.image_to_screen_axes .* (dilated_center - constricted_center);
    diameter_change_mm = dilated_diameter - constricted_diameter;
end
