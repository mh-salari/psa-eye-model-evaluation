function radius_mm = physical_radius(s, sg, pose, apparent_radius_mm)
%  The physical stop radius (mm) whose apparent pupil diameter in the pose is twice apparent_radius_mm, the stop
%  unshifted.
%
%  Each step rescales the physical radius by the ratio still missing, which converges because the apparent diameter is
%  close to proportional to the physical one.
    apparent_diameter_mm = 2 * apparent_radius_mm;
    radius_mm = apparent_radius_mm;
    for iteration = 1:s.pupil_diameter_max_iterations
        [~, simulated_mm] = gkamodeleye_scene.apparent_pupil(s, sg, pose, radius_mm, [0, 0]);
        if abs(simulated_mm - apparent_diameter_mm) <= s.pupil_diameter_tolerance_mm
            return
        end
        radius_mm = radius_mm * apparent_diameter_mm / simulated_mm;
    end
    error('the physical stop radius for an apparent pupil diameter of %.3f mm did not converge within %d iterations', ...
        apparent_diameter_mm, s.pupil_diameter_max_iterations);
end
