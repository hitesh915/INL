matrix = parser_arff('data/pima_diabetes.arff');
stdata = standarizer(matrix);

% -- K-Means for different K values
% ------------------------------------------------------

% -- PCA
% ------------------------------------------------------

% Run the PCA alhorithm for an eigenvalue threshold of 1
[dataAfterPCA transformedData eVectors eValues mostInfFeatures] = pca(stdata,1);

% Show most important features for the not discarded eigenvecotrs
mostInfFeatures

% Execute k_means for the original data once applied the PCA
execute_k_means(dataAfterPCA, 2, 5);
