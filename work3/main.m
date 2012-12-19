dataset1 = 'heart-statlog';
results1 = crossValidate(dataset);

% Save to file
fname = strcat('result_', dataset1, '.mat');
save(fname, 'results1');