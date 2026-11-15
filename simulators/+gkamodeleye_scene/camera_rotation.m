function rotation = camera_rotation(camera_world, aim_world, inter_eye_world, roll_deg)
%  gkaModelEye's camera rotation angles for the real camera's image axes.
%
%  The real camera is aimed at the midpoint between the eyes, with the line between the eyes along image x and image y
%  downward, and rolled about its view axis by roll_deg. Its image x runs toward the participant's right, which in a
%  right-handed camera looking at the participant is a mirror image, so the camera built here has image x toward the
%  participant's left and is otherwise the same. Its rotation matrix is decomposed into the three angles that
%  projectToImagePlane composes as Rz(rotation(3)) * Ry(-rotation(1)) * Rx(-rotation(2)) * diag([1 -1 -1]).
    view = (aim_world - camera_world) / norm(aim_world - camera_world);
    level_x = inter_eye_world - (inter_eye_world' * view) * view;
    level_x = level_x / norm(level_x);
    down = [0; -1; 0];
    level_y = down - (down' * view) * view - (down' * level_x) * level_x;
    level_y = level_y / norm(level_y);
    image_x = cosd(roll_deg) * level_x - sind(roll_deg) * level_y;
    image_y = sind(roll_deg) * level_x + cosd(roll_deg) * level_y;
    fick = [-image_x, image_y, view] * diag([1 -1 -1]);
    rotation = [asind(fick(3, 1)); -atan2d(fick(3, 2), fick(3, 3)); atan2d(fick(2, 1), fick(1, 1))];
end
