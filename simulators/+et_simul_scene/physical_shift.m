function physical_shift_m = physical_shift(s, c, eye, target, dilated_m, constricted_m, apparent_shift_m)
%  The eye-local physical pupil center shift (m) whose apparent shift at the target is apparent_shift_m (m, screen
%  axes).
%
%  The map from a physical to an apparent pupil center shift is treated as affine over this range. Its constant part is
%  the apparent shift without a physical shift, which the pupil diameter change produces on its own. Two probe
%  simulations give its linear part, which also carries the rotation between the eye's axes and the image's.
    zero = et_simul_scene.apparent_change(s, c, eye, target, dilated_m, constricted_m, [0; 0])';
    linear = zeros(2);
    for k = 1:2
        probe = zeros(2, 1);
        probe(k) = s.probe_m;
        linear(:, k) = (et_simul_scene.apparent_change(s, c, eye, target, dilated_m, constricted_m, probe)' ...
            - zero) / s.probe_m;
    end
    physical_shift_m = linear \ (apparent_shift_m - zero);
end
