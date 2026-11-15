function [shift_m, diameter_change_m] = apparent_change(s, c, eye, target, dilated_m, constricted_m, physical_shift_m)
%  Dilated minus constricted apparent pupil center (m at the eye, screen axes) and apparent pupil diameter (m) at one
%  target, the dilated pupil center moved by physical_shift_m.
    [dilated_center, dilated_diameter] = et_simul_scene.apparent_pupil(s, c, eye, target, dilated_m, physical_shift_m);
    [constricted_center, constricted_diameter] = et_simul_scene.apparent_pupil(s, c, eye, target, constricted_m, ...
        [0; 0]);
    shift_m = s.image_to_screen_axes .* (dilated_center - constricted_center);
    diameter_change_m = dilated_diameter - constricted_diameter;
end
