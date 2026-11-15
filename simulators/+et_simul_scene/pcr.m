function pcr_m = pcr(s, c, eye, target, physical_diameter_m, shift_m)
%  The P-CR, apparent pupil center minus corneal reflection (m at the eye, the real camera's image axes), at one target
%  for a physical pupil diameter and an eye-local shift of the pupil center (m).
    [ellipse, cr_px] = et_simul_scene.project(s, c, eye, target, physical_diameter_m, shift_m);
    if isempty(cr_px) || any(isnan(cr_px))
        error('et_simul found no corneal reflection at target %s', mat2str(target', 4));
    end
    pcr_m = et_simul_scene.m_per_px(c, eye) * (ellipse(1:2) - cr_px);
end
