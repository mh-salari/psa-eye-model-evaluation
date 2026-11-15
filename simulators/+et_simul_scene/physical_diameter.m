function physical_m = physical_diameter(s, c, eye, target, apparent_m)
%  The physical pupil diameter (m) whose apparent diameter at the target is apparent_m, the pupil center unshifted.
%
%  Each step rescales the physical diameter by the ratio still missing, which converges because the apparent diameter
%  is close to proportional to the physical one.
    physical_m = apparent_m;
    for iteration = 1:s.pupil_diameter_max_iterations
        [~, simulated_m] = et_simul_scene.apparent_pupil(s, c, eye, target, physical_m, [0; 0]);
        if abs(simulated_m - apparent_m) <= s.pupil_diameter_tolerance_m
            return
        end
        physical_m = physical_m * apparent_m / simulated_m;
    end
    error('the physical pupil diameter for an apparent %.3f mm did not converge within %d iterations', ...
        1e3 * apparent_m, s.pupil_diameter_max_iterations);
end
