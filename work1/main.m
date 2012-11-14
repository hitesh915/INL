matrix = parser_arff('data/breast-w.arff');
stdata = standarizer(matrix);

% -- K-Means for different K values
% ------------------------------------------------------

% Execute K-Means for K=2:5
%execute_algorithms(stdata, 2, 5, true, false);
%pause(1);

% -- PCA
% ------------------------------------------------------

% Get most relevant features with PCA and Apply K-Means to the
% transformed data
%execute_algorithms(stdata, 2, 5, true, true);
