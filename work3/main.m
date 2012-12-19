dataset1 = 'ionosphere';
results1 = crossValidate(dataset1);

% Save to file
fname = strcat('result_', dataset1, '.mat');
save(fname, 'results1');