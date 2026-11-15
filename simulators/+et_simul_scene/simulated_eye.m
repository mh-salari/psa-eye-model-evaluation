function eye = simulated_eye(s, eye_name, anatomy)
%  One eye's setup position (m) and the eye-local position of its pupil center relative to the optical axis (m), from
%  the anatomy's nasal and superior position (mm).
    eye.position = s.eye_positions(eye_name);
    eye.pupil_offset_m = 1e-3 * [s.nasal_sign(eye_name) * anatomy.off_axis_nasal_mm; anatomy.off_axis_superior_mm];
end
