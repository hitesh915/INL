clear

dataset = 'bal';
accuracies = zeros(1, 10);

for fold = 1:10
    [train, test] = parser_nfold(dataset, fold);
    accuracies(fold) = cbr( train, test, 3, 2 );
end

fprintf(strcat('Mean accuracy:\t\t', num2str(mean(accuracies)), '\n'));
fprintf(strcat('Accuracy standard dev.:\t', num2str(std(accuracies)), '\n'));
