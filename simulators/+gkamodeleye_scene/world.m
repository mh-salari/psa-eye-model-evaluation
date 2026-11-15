function position_world = world(position, eye_pos)
%  A setup position (x, height, distance) in gkaModelEye's world for one eye: the origin at that eye's unrotated
%  corneal apex, +x toward the participant's left, +y up and +z toward the screen.
    position_world = [-(position(1) - eye_pos(1)); position(2) - eye_pos(2); eye_pos(3) - position(3)];
end
