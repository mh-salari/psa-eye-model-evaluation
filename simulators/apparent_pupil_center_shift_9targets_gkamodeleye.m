% The apparent pupil center shift that gkaModelEye produces at the nine targets, next to the measured one.
%
% Each eye is simulated twice at every target: constricted, and dilated with its aperture stop moved by a physical
% pupil center shift. The physical stop diameters and the physical pupil center shift are the measured apparent pupil
% diameters and pupil center shift at the center target, the target the magnitude and direction change are measured
% from. They are the same at every target, so the eye orientation is all that differs between targets. At each target
% the eye takes the pose that places its line of sight on the target.
%
% With solve_physical set to true, they are instead inferred: the physical stop diameters and the physical pupil center
% shift for which the simulated apparent ones at the center target equal the measured ones. With eye_parameters set to
% a fit_eye_parameters_gkamodeleye.m table of the same camera, each eye takes its fitted anatomy instead of
% gkaModelEye's own eye, and any eye the fit does not cover is left out and named in the output.
%
% The simulated apparent pupil center shift and apparent pupil diameter change at a target are the dilated minus the
% constricted pupil center and pupil diameter in the two simulated images, in mm at the eye, the diameter being the mean
% of the fitted ellipse's two axes as in the recordings. The measured and the simulated apparent pupil center shift at a
% target are each divided by their own apparent pupil diameter change at that target, measured or simulated.
%
% The camera has the real camera's image axes: it is aimed at the midpoint between the eyes, with the line between
% the eyes along image x and image y downward, and then rolled about its view axis by the roll measured in the
% recordings.
%
% Runs on the camera set by the `camera` variable, 'cam2' (eye-level, the default) or 'cam1' (bottom-left), infers the
% physical values when `solve_physical` is true, and takes the fitted anatomy from the `eye_parameters` table, a path
% relative to simulators/, where run() executes the script:
%
%   /Applications/MATLAB_R2026a.app/bin/matlab -batch "camera = 'cam1'; solve_physical = true; eye_parameters = 'output/eye_parameters_gkamodeleye_bottom_left.csv'; run('simulators/apparent_pupil_center_shift_9targets_gkamodeleye.m')"
%
% Writes apparent_pupil_center_shift_9targets_gkamodeleye{_run}{_cam1}.csv to simulators/output/, where the run names
% the fit, followed by "physical" for solve_physical. Each row also carries the simulated apparent pupil diameter
% change, the eye's two physical pupil diameters and its simulated P-CR (pupil center minus glint) in both conditions,
% in mm at the eye and in the camera's image axes.

if ~exist('camera', 'var'), camera = 'cam2'; end
if ~exist('solve_physical', 'var'), solve_physical = false; end
if ~exist('eye_parameters', 'var'), eye_parameters = ''; end
result_file_camera_suffix = containers.Map({'cam2', 'cam1'}, {'', '_cam1'});
fit_file_prefix = 'eye_parameters_gkamodeleye_';

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, 'gkaModelEye')));
addpath(here);
s = gkamodeleye_scene.setup();

run_parts = {};
if ~isempty(eye_parameters)
    [~, fit_stem] = fileparts(eye_parameters);
    if ~startsWith(fit_stem, fit_file_prefix)
        error('%s is not a fit_eye_parameters_gkamodeleye.m table', eye_parameters);
    end
    run_parts{end + 1} = extractAfter(fit_stem, fit_file_prefix);
end
if solve_physical, run_parts{end + 1} = 'physical'; end
run_suffix = strjoin(strcat('_', run_parts), '');

% ---------------------------------------------------------------------------
% Input
% ---------------------------------------------------------------------------

input_path = fullfile(here, '..', 'measured', 'output', 'simulator_input.csv');
if ~isfile(input_path)
    error('%s is missing; run measured/simulator_input.py', input_path);
end
options = detectImportOptions(input_path, 'TextType', 'char');
options = setvartype(options, {'camera', 'participant', 'eye', 'target_label'}, 'char');
measured = readtable(input_path, options);
measured = measured(strcmp(measured.camera, camera), :);
if isempty(measured)
    error('no %s rows in %s', camera, input_path);
end
roll_deg = unique(measured.camera_roll_deg);
if numel(roll_deg) ~= 1
    error('%s has %d different camera rolls in %s', camera, numel(roll_deg), input_path);
end

eyes = unique(strcat(measured.participant, '|', measured.eye));
population = gkamodeleye_scene.population_anatomy(s);
if isempty(eye_parameters)
    anatomies = containers.Map(eyes, repmat({population}, size(eyes)), 'UniformValues', false);
else
    anatomies = fitted_anatomy(eye_parameters, camera, fieldnames(population));
    for name = setdiff(eyes, keys(anatomies))'
        fprintf('left out: %s has no fitted anatomy in %s\n', strrep(name{1}, '|', ' '), eye_parameters);
    end
    eyes = intersect(eyes, keys(anatomies));
end

% ---------------------------------------------------------------------------
% Simulation
% ---------------------------------------------------------------------------

participant_column = cell(0, 1); eye_column = cell(0, 1); target_column = zeros(0, 1); label_column = cell(0, 1);
measured_rate = zeros(0, 2); simulated_rate = zeros(0, 2); simulated_diameter_change_column = zeros(0, 1);
physical_dilated_column = zeros(0, 1); physical_constricted_column = zeros(0, 1);
pcr_constricted_column = zeros(0, 2); pcr_dilated_column = zeros(0, 2);
for k = 1:numel(eyes)
    parts = split(eyes{k}, '|');
    participant = parts{1};
    eye_name = parts{2};
    eye_rows = sortrows(measured(strcmp(measured.participant, participant) & strcmp(measured.eye, eye_name), :), ...
        'target');
    if ~isequal(eye_rows.target', 0:8)
        error('%s %s %s has targets %s, expected all nine', camera, participant, eye_name, mat2str(eye_rows.target'));
    end
    sg = gkamodeleye_scene.scene(s, eye_name, camera, roll_deg, anatomies(eyes{k}));
    eye_poses = gkamodeleye_scene.fixation_poses(s, sg, eye_name);

    shift_mm = [eye_rows.apparent_pupil_center_shift_x_mm, eye_rows.apparent_pupil_center_shift_y_mm];
    diameter_change_mm = eye_rows.apparent_pupil_diameter_dilated_mm - eye_rows.apparent_pupil_diameter_constricted_mm;
    dilated_radius_mm = eye_rows.apparent_pupil_diameter_dilated_mm(s.center_row) / 2;
    constricted_radius_mm = eye_rows.apparent_pupil_diameter_constricted_mm(s.center_row) / 2;
    % The eye's horizontal axis runs toward the participant's left, as gkaModelEye's world x does.
    stop_shift_mm = [-1, 1] .* shift_mm(s.center_row, :);
    if solve_physical
        center_pose = eye_poses(s.center_row, :);
        dilated_radius_mm = gkamodeleye_scene.physical_radius(s, sg, center_pose, dilated_radius_mm);
        constricted_radius_mm = gkamodeleye_scene.physical_radius(s, sg, center_pose, constricted_radius_mm);
        stop_shift_mm = gkamodeleye_scene.physical_stop_shift(s, sg, center_pose, dilated_radius_mm, ...
            constricted_radius_mm, shift_mm(s.center_row, :));
    end

    simulated_mm = zeros(9, 2);
    simulated_diameter_change_mm = zeros(9, 1);
    pcr_constricted_mm = zeros(9, 2);
    pcr_dilated_mm = zeros(9, 2);
    for t = 1:9
        [simulated_mm(t, :), simulated_diameter_change_mm(t)] = gkamodeleye_scene.apparent_change(s, sg, ...
            eye_poses(t, :), dilated_radius_mm, constricted_radius_mm, stop_shift_mm);
        pcr_constricted_mm(t, :) = gkamodeleye_scene.pcr(s, sg, eye_poses(t, :), constricted_radius_mm, [0, 0]);
        pcr_dilated_mm(t, :) = gkamodeleye_scene.pcr(s, sg, eye_poses(t, :), dilated_radius_mm, stop_shift_mm);
    end

    participant_column = [participant_column; repmat({participant}, 9, 1)]; %#ok<AGROW>
    eye_column = [eye_column; repmat({eye_name}, 9, 1)]; %#ok<AGROW>
    target_column = [target_column; eye_rows.target]; %#ok<AGROW>
    label_column = [label_column; s.target_labels']; %#ok<AGROW>
    measured_rate = [measured_rate; shift_mm ./ diameter_change_mm]; %#ok<AGROW>
    simulated_rate = [simulated_rate; simulated_mm ./ simulated_diameter_change_mm]; %#ok<AGROW>
    simulated_diameter_change_column = [simulated_diameter_change_column; simulated_diameter_change_mm]; %#ok<AGROW>
    physical_dilated_column = [physical_dilated_column; repmat(2 * dilated_radius_mm, 9, 1)]; %#ok<AGROW>
    physical_constricted_column = [physical_constricted_column; repmat(2 * constricted_radius_mm, 9, 1)]; %#ok<AGROW>
    pcr_constricted_column = [pcr_constricted_column; pcr_constricted_mm]; %#ok<AGROW>
    pcr_dilated_column = [pcr_dilated_column; pcr_dilated_mm]; %#ok<AGROW>

    measured_center_rate = shift_mm(s.center_row, :) / diameter_change_mm(s.center_row);
    simulated_center_rate = simulated_mm(s.center_row, :) / simulated_diameter_change_mm(s.center_row);
    cosine = dot(measured_center_rate, simulated_center_rate) / ...
        (norm(measured_center_rate) * norm(simulated_center_rate));
    fprintf(['%s %-9s physical pupil diameter dilated %.2f mm, constricted %.2f mm; center target, simulated ' ...
        'against measured per mm of pupil diameter change: magnitude ratio %.3f, angle %.1f deg\n'], participant, ...
        eye_name, 2 * dilated_radius_mm, 2 * constricted_radius_mm, ...
        norm(simulated_center_rate) / norm(measured_center_rate), acosd(min(1, max(-1, cosine))));
end

result = table(participant_column, eye_column, target_column, label_column, ...
    measured_rate(:, 1), measured_rate(:, 2), simulated_rate(:, 1), simulated_rate(:, 2), ...
    simulated_diameter_change_column, physical_dilated_column, physical_constricted_column, ...
    pcr_constricted_column(:, 1), pcr_constricted_column(:, 2), pcr_dilated_column(:, 1), pcr_dilated_column(:, 2), ...
    'VariableNames', {'participant', 'eye', 'target', 'target_label', ...
        'measured_apparent_pupil_center_shift_x_mm_per_mm', 'measured_apparent_pupil_center_shift_y_mm_per_mm', ...
        'simulated_apparent_pupil_center_shift_x_mm_per_mm', 'simulated_apparent_pupil_center_shift_y_mm_per_mm', ...
        'simulated_apparent_pupil_diameter_change_mm', 'physical_pupil_diameter_dilated_mm', ...
        'physical_pupil_diameter_constricted_mm', 'simulated_pcr_constricted_x_mm', 'simulated_pcr_constricted_y_mm', ...
        'simulated_pcr_dilated_x_mm', 'simulated_pcr_dilated_y_mm'});
out_dir = fullfile(here, 'output');
if ~isfolder(out_dir), mkdir(out_dir); end
result_path = fullfile(out_dir, ['apparent_pupil_center_shift_9targets_gkamodeleye' run_suffix ...
    result_file_camera_suffix(camera) '.csv']);
writetable(result, result_path);
fprintf('wrote %s\n', result_path);


% ---------------------------------------------------------------------------
% Local functions
% ---------------------------------------------------------------------------

function anatomies = fitted_anatomy(path, camera, fitted)
%  {'participant|eye': anatomy} from a fit_eye_parameters_gkamodeleye.m table fitted to the camera.
    if ~isfile(path)
        error('%s is missing; run fit_eye_parameters_gkamodeleye.m for %s', path, camera);
    end
    options = detectImportOptions(path, 'TextType', 'char');
    options = setvartype(options, {'participant', 'eye', 'camera'}, 'char');
    fit = readtable(path, options);
    if ~all(strcmp(fit.camera, camera))
        error('%s was not fitted to %s', path, camera);
    end
    anatomies = containers.Map();
    for r = 1:height(fit)
        for p = 1:numel(fitted)
            anatomy.(fitted{p}) = fit.(fitted{p})(r);
        end
        anatomies([fit.participant{r} '|' fit.eye{r}]) = anatomy;
    end
end
