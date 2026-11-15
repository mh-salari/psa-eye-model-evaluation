function sg = scene(s, eye_name, camera, roll_deg, anatomy)
%  gkaModelEye's scene for one eye with the given anatomy, seen by one camera under the setup's illuminator.
%
%  gkaModelEye's world is centered on the eye, so each eye has its own scene. The anatomy's corneal apex radius sets the
%  mean corneal power, and both principal meridians are shifted by the same power, so gkaModelEye's own astigmatism is
%  kept. The aperture stop is centered at the anatomy's nasal and superior position relative to the optical axis.
    import gkamodeleye_scene.world
    eye_pos = s.eye_positions(eye_name);
    camera_world = world(s.camera_positions(camera), eye_pos);
    midpoint = (s.eye_positions('left_eye') + s.eye_positions('right_eye')) / 2;
    inter_eye_world = world(s.eye_positions('right_eye'), eye_pos) - world(s.eye_positions('left_eye'), eye_pos);
    rotation = gkamodeleye_scene.camera_rotation(camera_world, world(midpoint, eye_pos), inter_eye_world, roll_deg);
    % gkaModelEye places the glint source at the camera position plus this offset, without rotating it.
    glint_source_relative = world(s.illuminator_position, eye_pos) - camera_world;
    mean_power = 1000 * (s.keratometric_index - 1) / anatomy.anterior_radius;
    kvals = s.default_kvals + (mean_power - mean(s.default_kvals));
    sg = createSceneGeometry('eyeLaterality', strrep(eye_name, '_eye', ''), ...
        'cameraTranslation', camera_world, 'cameraRotation', rotation, ...
        'cameraGlintSourceRelative', glint_source_relative, 'kvals', kvals);
    sg.eye.stop.center(2:3) = sg.eye.stop.center(2:3) + ...
        [s.nasal_sign(eye_name) * anatomy.off_axis_nasal_mm, anatomy.off_axis_superior_mm];
end
