function poses = fixation_poses(s, sg, eye_name)
%  The eye poses [azimuth, elevation, torsion, stop radius] that place the eye's line of sight on each of the nine
%  targets, one row per target.
%
%  gkaModelEye's line of sight runs from the fovea through the center of the entrance pupil, so the poses depend on the
%  stop position as well as on the cornea. They are found with the stop unshifted by a pupil center shift and used in
%  both the dilated and constricted conditions, since the eye orientation at a target is the same in both.
    eye_pos = s.eye_positions(eye_name);
    node = sg.eye.landmarks.incidentNode.coords;
    poses = zeros(size(s.targets, 1), 4);
    for t = 1:size(s.targets, 1)
        target_world = gkamodeleye_scene.world([s.targets(t, :), 0], eye_pos);
        direction = convertWorldToEyeCoord(target_world) - node;
        [horizontal_deg, vertical_deg] = quadric.rayToAngles(quadric.normalizeRay([node', direction']));
        poses(t, :) = calcFixationPose(sg.eye, [horizontal_deg, vertical_deg], norm(direction));
    end
end
