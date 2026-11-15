function c = setup_camera(s, camera, roll_deg)
%  The camera at its setup position with the real camera's image axes.
%
%  It is aimed at the midpoint between the eyes, with the line between the eyes along image x and image y downward,
%  and then rolled about its view axis by roll_deg, so that the line between the eyes lies at roll_deg in the image
%  as it does in the recordings. The image axes are set as the camera's x and y axes, which et_simul projects onto.
    camera_pos = s.camera_positions(camera);
    left_eye_pos = s.eye_positions('left_eye');
    right_eye_pos = s.eye_positions('right_eye');
    view = (left_eye_pos + right_eye_pos) / 2 - camera_pos;
    view = view / norm(view);
    inter_eye = right_eye_pos - left_eye_pos;
    level_x = inter_eye - (inter_eye' * view) * view;
    level_x = level_x / norm(level_x);
    level_y = cross(view, level_x);
    image_x = cosd(roll_deg) * level_x - sind(roll_deg) * level_y;
    image_y = sind(roll_deg) * level_x + cosd(roll_deg) * level_y;
    c = camera_make();
    % et_simul's camera looks along its negative z axis.
    c.trans(1:3, 1:3) = [image_x, image_y, -view];
    c.trans(1:3, 4) = camera_pos;
    c.rest_trans = c.trans;
end
