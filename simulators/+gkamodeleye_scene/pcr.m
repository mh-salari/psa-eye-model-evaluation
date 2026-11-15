function pcr_mm = pcr(s, sg, pose, stop_radius_mm, stop_shift_mm)
%  The P-CR, apparent pupil center minus glint (mm at the eye, the real camera's image axes), in one pose for a physical
%  stop radius and a shift of the stop center (mm).
    [ellipse, glint_px] = gkamodeleye_scene.project(sg, pose, stop_radius_mm, stop_shift_mm, s.stop_perimeter_points);
    if isempty(glint_px) || any(isnan(glint_px(:)))
        error('gkaModelEye found no glint in pose %s', mat2str(pose, 4));
    end
    pcr_mm = s.image_to_camera_axes .* (gkamodeleye_scene.mm_per_px(sg) * (ellipse(1:2) - glint_px(1, :)));
end
