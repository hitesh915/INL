function [ train_matrix, test_matrix ] = parser_nfold(dataset, fold)
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
    [train_matrix, classes] = parser_arff(strcat(base, 'train.arff'));
    [test_matrix, classes] = parser_arff(strcat(base, 'test.arff'));
    
    % Show classes glossary
    if fold == 1
        clist = 'Classes glossay:';
        csize = size(classes, 2);
        for i=1:csize
            cname = cell2mat(classes(1, i));
            clist = strcat(clist, '\n', cname, '\t=>\t', num2str(i));
        end

        fprintf(strcat(clist, '\n\n'));
    end
end

