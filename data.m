% Load your Simulink model
model_name = 'test1';
load_system(model_name)
% Define the number of data points
num_data_points = 1000;

ir_min = 100;            // ir = iridance 
                         // t = ambient temp
ir_max = 1500;
t_min = 15;
t_max = 35;

output_csv_file = 'data_set.csv';

if ~isfile(output_csv_file)
header = {'Ir', 'T', 'Power'};
writetable(cell2table(cell(0,3), 'VariableNames', header), output_csv_file);
end
% Loop over data points
for i = 1:num_data_points
ir = ir_min + rand() * (ir_max - ir_min); % Generate a random value within the range
t = t_min + rand() * (t_max - t_min); % Generate a random value within the range
% Set the parameter values
set_param([model_name '/Constant'], 'Value', num2str(ir));
set_param([model_name '/Constant1'], 'Value', num2str(t));
% Start the simulation
simOut = sim(model_name);
% Access the logged signal data loggedData = simOut.get('loggedData').signals.values;
% Append the input, t, and last output to the CSV file
data = [ir, t, loggedData(end)];
dlmwrite(output_csv_file, data, '-append');
end
% Close the Simulink model
close_system(model_name, 0);
