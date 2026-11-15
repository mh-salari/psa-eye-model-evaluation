function s = setup()
%  The recording setup in mm, as (x toward the participant's right, height above the screen center, distance from the
%  screen), the image axes, and the settings of the physical pupil searches. The positions are experiment_setup.py's. An
%  eye's position is its corneal apex when the eye is unrotated.
    s.eye_positions = containers.Map({'left_eye', 'right_eye'}, {[-30, 100, 814], [30, 100, 814]});
    s.camera_positions = containers.Map({'cam1', 'cam2'}, {[-160, -95, 315], [0, 105, 195]});
    s.illuminator_position = [75, -97.5, 435];
    % The nine HV9 targets on the screen plane (distance 0), as (x, height), in target index order.
    s.targets = [ ...
           0.000     0.000; ...
           0.000   124.298; ...
           0.000  -124.021; ...
        -163.836     0.000; ...
         163.559     0.000; ...
        -122.877    93.293; ...
         122.600    93.293; ...
        -122.877   -93.016; ...
         122.600   -93.016];
    s.target_labels = {'Center', 'Top-C', 'Bot-C', 'Left', 'Right', 'Top-L', 'Top-R', 'Bot-L', 'Bot-R'};
    s.center_row = 1;
    % Outline points of the aperture stop that the entrance pupil ellipse is fitted to.
    s.stop_perimeter_points = 36;
    % gkaModelEye's image x runs toward the participant's left and its image y downward. Negating x gives the real
    % camera's image axes, and negating both gives the screen axes.
    s.image_to_camera_axes = [-1, 1];
    s.image_to_screen_axes = [-1, -1];
    % The physical stop diameter search stops once the simulated apparent pupil diameter at the center target is this
    % close to the measured one, and fails loudly if it has not got there within the iteration limit.
    s.pupil_diameter_tolerance_mm = 1e-4;
    s.pupil_diameter_max_iterations = 20;
    % Eye-local step of the two probe simulations that measure how a physical pupil center shift appears in the image.
    s.probe_mm = 0.05;
    % gkaModelEye's default corneal powers (D) in its two principal meridians, and the keratometric index it converts a
    % power K to a radius of curvature with, R = 1000 * (n - 1) / K mm.
    s.default_kvals = [43.399, 44.33653846];
    s.keratometric_index = 1.3375;
    % Sign of the nasal direction along the aperture stop's horizontal coordinate, which runs toward the participant's
    % left in both eyes.
    s.nasal_sign = containers.Map({'left_eye', 'right_eye'}, {-1, 1});
end
