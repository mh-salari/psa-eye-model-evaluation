% Fit each eye's anatomy in gkaModelEye to that eye's measured P-CR on one camera, in the constricted condition.
%
% The simulator otherwise runs gkaModelEye's own eye. This fits, per eye, the corneal apex radius and the nasal and
% superior position of the pupil center relative to the optical axis, within published human bounds, so that the
% simulated constricted P-CR matches the measured one at all nine targets. The corneal apex radius sets gkaModelEye's
% mean corneal power, with both principal meridians shifted by the same power, so its own astigmatism is kept. For
% every anatomy the fit tries, the eye poses are gkaModelEye's fixation poses for that eye, and the physical stop radius
% is the one whose apparent pupil diameter at the center target equals the measured constricted one, inferred as in the
% nine-target runs with solve_physical.
%
% Only the constricted condition is fitted: the apparent pupil center shift lives in the dilated-minus-constricted
% difference, so a fit to the constricted P-CR cannot tune it.
%
% Each fit matches one camera's P-CR at the nine targets, 9 x 2 = 18 constraints for the three fitted parameters, so
% the simulations of a camera use eyes fitted to that camera's view alone. Eyes the camera did not record are left out
% and named in the output.
%
% Runs on the camera set by the `camera` variable, 'cam2' (eye-level, the default) or 'cam1' (bottom-left):
%
%   /Applications/MATLAB_R2026a.app/bin/matlab -batch "camera = 'cam1'; run('simulators/fit_eye_parameters_gkamodeleye.m')"
%
% Writes eye_parameters_gkamodeleye_<camera>.csv to simulators/output/, <camera> being eye_level or bottom_left.

if ~exist('camera', 'var'), camera = 'cam2'; end
fit_file_camera = containers.Map({'cam2', 'cam1'}, {'eye_level', 'bottom_left'});

% Published human mean and SD of each fitted parameter. Each is bounded to its mean +- bound_sds SD, so a fitted value
% at a bound means the fit wants anatomy outside the range reported for human eyes.
%   anterior_radius: corneal apex radius, Table 2.2 of Atchison, D. A. (2023). Optics of the Human Eye (2nd ed.).
%       CRC Press.
%   off_axis_nasal_mm, off_axis_superior_mm: the pupil center's offset, photopic row (the fit uses the constricted
%       pupil), Table 1 of Yang, Y., Thompson, K., & Burns, S. A. (2002). Pupil location under mesopic, photopic, and
%       pharmacologically dilated conditions. Investigative Ophthalmology & Visual Science, 43(7), 2508-2512.
fitted = {'anterior_radius', 'off_axis_nasal_mm', 'off_axis_superior_mm'};
human_mean = [7.79, 0.202, 0.087];
human_sd = [0.25, 0.120, 0.151];
bound_sds = 3;
lower = human_mean - bound_sds * human_sd;
upper = human_mean + bound_sds * human_sd;
% A fitted value within this distance of a bound, in the parameter's units, counts as having reached it.
rail_tolerance = 1e-3;
% Relative finite-difference step of the fit. gkaModelEye's fixation pose and glint searches stop at their own
% tolerances, so the simulated P-CR follows the parameters smoothly only over steps well above those tolerances.
finite_difference_step = 1e-4;

here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, 'gkaModelEye')));
addpath(here);
s = gkamodeleye_scene.setup();
population = gkamodeleye_scene.population_anatomy(s);
start = cellfun(@(name) population.(name), fitted);

% ---------------------------------------------------------------------------
% Input
% ---------------------------------------------------------------------------

roll_deg = camera_roll(fullfile(here, '..', 'measured', 'output', 'simulator_input.csv'), camera);
fit_input_path = fullfile(here, '..', 'measured', 'output', 'fit_input.csv');
if ~isfile(fit_input_path)
    error('%s is missing; run measured/simulator_input.py', fit_input_path);
end
options = detectImportOptions(fit_input_path, 'TextType', 'char');
options = setvartype(options, {'camera', 'participant', 'eye', 'target_label'}, 'char');
fit_input = readtable(fit_input_path, options);
all_eyes = unique(strcat(fit_input.participant, '|', fit_input.eye));
fit_input = fit_input(strcmp(fit_input.camera, camera), :);
eyes = unique(strcat(fit_input.participant, '|', fit_input.eye));
for name = setdiff(all_eyes, eyes)'
    fprintf('left out: %s is not recorded by %s\n', strrep(name{1}, '|', ' '), camera);
end

% ---------------------------------------------------------------------------
% Fit
% ---------------------------------------------------------------------------

fprintf('Fitting %d parameters for %d eyes on %s (%d constraints each)\n', numel(fitted), numel(eyes), camera, ...
    2 * size(s.targets, 1));
fit_options = optimoptions('lsqnonlin', 'Display', 'off', 'FiniteDifferenceStepSize', finite_difference_step, ...
    'OutputFcn', @report_iteration);
participant_column = cell(numel(eyes), 1); eye_column = cell(numel(eyes), 1);
values = zeros(numel(eyes), numel(fitted)); rails = cell(numel(eyes), numel(fitted));
rms_before_mm = zeros(numel(eyes), 1); rms_after_mm = zeros(numel(eyes), 1);
started = tic;
for k = 1:numel(eyes)
    parts = split(eyes{k}, '|');
    participant_column{k} = parts{1};
    eye_column{k} = parts{2};
    eye_rows = sortrows(fit_input(strcmp(fit_input.participant, parts{1}) & strcmp(fit_input.eye, parts{2}), :), ...
        'target');
    if ~isequal(eye_rows.target', 0:8)
        error('%s %s %s has targets %s, expected all nine', camera, parts{1}, parts{2}, mat2str(eye_rows.target'));
    end
    measured_pcr_mm = [eye_rows.pcr_constricted_x_mm, eye_rows.pcr_constricted_y_mm];
    apparent_diameter_mm = eye_rows.apparent_pupil_diameter_constricted_mm(s.center_row);
    objective = @(vector) pcr_error(s, fitted, vector, parts{2}, camera, roll_deg, measured_pcr_mm, ...
        apparent_diameter_mm);

    rms_before_mm(k) = rms_error(objective(start));
    [values(k, :), ~, residual] = lsqnonlin(objective, start, lower, upper, fit_options);
    rms_after_mm(k) = rms_error(residual);
    for p = 1:numel(fitted)
        rails{k, p} = railed(values(k, p), lower(p), upper(p), rail_tolerance);
    end

    summary = strjoin(arrayfun(@(p) sprintf('%s=%+.3f%s', fitted{p}, values(k, p), ...
        repmat('!', 1, ~isempty(rails{k, p}))), 1:numel(fitted), 'UniformOutput', false), '  ');
    elapsed = toc(started);
    fprintf('[%d/%d] %s %-9s rms %6.1f -> %6.1f um   %s   (%.0f min, about %.0f min left)\n', k, numel(eyes), ...
        parts{1}, parts{2}, 1000 * rms_before_mm(k), 1000 * rms_after_mm(k), summary, elapsed / 60, ...
        elapsed / k * (numel(eyes) - k) / 60);
end

% ---------------------------------------------------------------------------
% Output
% ---------------------------------------------------------------------------

result = [table(participant_column, eye_column, repmat({camera}, numel(eyes), 1), ...
        'VariableNames', {'participant', 'eye', 'camera'}), ...
    array2table(values, 'VariableNames', fitted), ...
    cell2table(rails, 'VariableNames', strcat(fitted, '_railed')), ...
    table(round(1000 * rms_before_mm, 2), round(1000 * rms_after_mm, 2), ...
        'VariableNames', {'rms_before_um', 'rms_after_um'})];
out_dir = fullfile(here, 'output');
if ~isfolder(out_dir), mkdir(out_dir); end
result_path = fullfile(out_dir, ['eye_parameters_gkamodeleye_' fit_file_camera(camera) '.csv']);
writetable(result, result_path);
fprintf('\nwrote %s\n', result_path);

fprintf('\nP-CR error over %d eyes, median: %.1f -> %.1f um\n', numel(eyes), 1000 * median(rms_before_mm), ...
    1000 * median(rms_after_mm));
for p = 1:numel(fitted)
    fprintf('  %-22s median %+.3f  range %+.3f..%+.3f  bounds %+.3f..%+.3f  railed %d/%d\n', fitted{p}, ...
        median(values(:, p)), min(values(:, p)), max(values(:, p)), lower(p), upper(p), ...
        sum(~cellfun(@isempty, rails(:, p))), numel(eyes));
end


% ---------------------------------------------------------------------------
% Local functions
% ---------------------------------------------------------------------------

function roll_deg = camera_roll(input_path, camera)
%  The camera's roll about its view axis measured in the recordings (deg).
    if ~isfile(input_path)
        error('%s is missing; run measured/simulator_input.py', input_path);
    end
    options = detectImportOptions(input_path, 'TextType', 'char');
    options = setvartype(options, {'camera', 'participant', 'eye', 'target_label'}, 'char');
    measured = readtable(input_path, options);
    roll_deg = unique(measured.camera_roll_deg(strcmp(measured.camera, camera)));
    if numel(roll_deg) ~= 1
        error('%s has %d different camera rolls in %s', camera, numel(roll_deg), input_path);
    end
end


function residual = pcr_error(s, fitted, vector, eye_name, camera, roll_deg, measured_pcr_mm, apparent_diameter_mm)
%  Simulated minus measured constricted P-CR (mm) at the nine targets, as a column of (x, y) pairs in target order.
    anatomy = cell2struct(num2cell(vector(:)), fitted(:), 1);
    sg = gkamodeleye_scene.scene(s, eye_name, camera, roll_deg, anatomy);
    poses = gkamodeleye_scene.fixation_poses(s, sg, eye_name);
    radius_mm = gkamodeleye_scene.physical_radius(s, sg, poses(s.center_row, :), apparent_diameter_mm / 2);
    simulated_mm = zeros(size(poses, 1), 2);
    for t = 1:size(poses, 1)
        simulated_mm(t, :) = gkamodeleye_scene.pcr(s, sg, poses(t, :), radius_mm, [0, 0]);
    end
    residual = reshape((simulated_mm - measured_pcr_mm)', [], 1);
end


function value = rms_error(residual)
%  Root-mean-square P-CR error in mm from a column of (x, y) pairs.
    value = sqrt(mean(sum(reshape(residual, 2, []) .^ 2, 1)));
end


function side = railed(value, low, high, tolerance)
%  Which bound a fitted value has reached, or '' when it lies inside its range.
    side = '';
    if value - low <= tolerance
        side = 'lower';
    elseif high - value <= tolerance
        side = 'upper';
    end
end


function stop = report_iteration(~, optim_values, state)
%  Streams the fit's progress within an eye: the iteration and its RMS P-CR error over the nine targets.
    stop = false;
    if strcmp(state, 'iter')
        fprintf('    iteration %2d  rms %6.1f um\n', optim_values.iteration, 1000 * rms_error(optim_values.residual));
    end
end
