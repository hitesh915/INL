function [ train_features, train_classes, test_features, test_classes ] = parser_nfold(dataset, fold)
    % PARSER_NFOLD parses both the train and test datasets for a specific
    % fold of the given dataset.
    %
    % INPUTS:
    %   dataset = Name of the dataset to parse
    %   fold = Value between 1 and 10 of the fold to parse
    %
    % OUTPUTS:
    %   train_features = Matrix of features for the training set
    %   train_classes = Classes assigned to the training individuals
    %   test_features = Matrix of features for the test set
    %   test_classes = Classes assigned to the test individuals
    
    base = strcat('data/', dataset, '/', dataset, '.fold.00000', num2str(fold-1), '.');
    [train_features train_classes] = parser_arff(strcat(base, 'train.arff'));
    [test_features test_classes] = parser_arff(strcat(base, 'test.arff'));
end

