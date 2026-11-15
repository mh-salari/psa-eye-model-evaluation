function stop_shift_mm = physical_stop_shift(s, sg, pose, dilated_radius_mm, constricted_radius_mm, ...
        apparent_shift_mm)
%  The shift of the stop center (mm, eye axes) whose apparent pupil center shift in the pose is apparent_shift_mm (mm,
%  screen axes).
%
%  The map from a physical to an apparent pupil center shift is treated as affine over this range. Its constant part is
%  the apparent shift without a physical shift, which the pupil diameter change produces on its own. Two probe
%  simulations give its linear part, which also carries the rotation between the eye's axes and the image's.
    zero = gkamodeleye_scene.apparent_change(s, sg, pose, dilated_radius_mm, constricted_radius_mm, [0, 0])';
    linear = zeros(2);
    for k = 1:2
        probe = zeros(1, 2);
        probe(k) = s.probe_mm;
        linear(:, k) = (gkamodeleye_scene.apparent_change(s, sg, pose, dilated_radius_mm, constricted_radius_mm, ...
            probe)' - zero) / s.probe_mm;
    end
    stop_shift_mm = (linear \ (apparent_shift_mm' - zero))';
end
