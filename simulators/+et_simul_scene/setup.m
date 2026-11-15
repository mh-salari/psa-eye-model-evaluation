function s = setup()
%  The recording setup in meters, the image axes, and the settings of the physical pupil searches.
%
%  The positions are experiment_setup.py's, in et_simul's world axes as its example.m uses them: x toward the
%  participant's right, y away from the screen, z height above the screen center. An eye's position is its corneal
%  apex when the eye is unrotated.
    global FEAT_FOVEA_DISPLACEMENT
    % The offset between the optical and the visual axis.
    FEAT_FOVEA_DISPLACEMENT = 1;

    s.eye_positions = containers.Map({'left_eye', 'right_eye'}, {[-30e-3; 814e-3; 100e-3], [30e-3; 814e-3; 100e-3]});
    s.camera_positions = containers.Map({'cam1', 'cam2'}, {[-160e-3; 315e-3; -95e-3], [0; 195e-3; 105e-3]});
    s.illuminator_position = [75e-3; 435e-3; -97.5e-3];
    % The nine HV9 targets on the screen plane (y = 0), in target index order.
    s.targets = 1e-3 * [ ...
           0.000  0.000    0.000; ...
           0.000  0.000  124.298; ...
           0.000  0.000 -124.021; ...
        -163.836  0.000    0.000; ...
         163.559  0.000    0.000; ...
        -122.877  0.000   93.293; ...
         122.600  0.000   93.293; ...
        -122.877  0.000  -93.016; ...
         122.600  0.000  -93.016]';
    s.target_labels = {'Center', 'Top-C', 'Bot-C', 'Left', 'Right', 'Top-L', 'Top-R', 'Bot-L', 'Bot-R'};
    s.center_column = 1;
    % et_simul's image has the real camera's image axes. Image y runs downward and screen y upward, so the vertical
    % component is negated to reach the screen axes.
    s.image_to_screen_axes = [1, -1];
    % Points on the refracted pupil boundary that the pupil ellipse is fitted to.
    s.pupil_boundary_points = 360;
    % et_simul's default corneal radius of curvature, which eye_make scales the whole front of the eye by.
    s.cornea_radius_m = 7.98e-3;
    % Eye-local x along world x, eye-local y along world z (height) and eye-local z along world y, so that the eye,
    % which looks along its negative z axis, faces the screen.
    s.rest_orientation = [1 0 0; 0 0 1; 0 1 0];
    % Sign of the nasal direction along eye-local x, which runs toward the participant's right in both eyes.
    s.nasal_sign = containers.Map({'left_eye', 'right_eye'}, {1, -1});
    % The physical pupil diameter search stops once the simulated apparent pupil diameter at the center target is this
    % close to the measured one, and fails loudly if it has not got there within the iteration limit.
    s.pupil_diameter_tolerance_m = 1e-7;
    s.pupil_diameter_max_iterations = 20;
    % Eye-local step of the two probe simulations that measure how a physical pupil center shift appears in the image.
    s.probe_m = 0.05e-3;
end
