%[features, classes] = parser_arff('data/bal/bal.fold.000000.test.arff');

dataset = 'breast-w';

accuracies = zeros(1, 10);

for fold = 1:10
    [train, test] = parser_nfold('dataset', fold);
    accuracies(i) = cbr( train, test, 3, 2 );
end

mean(accuracies)
std(accuracies)