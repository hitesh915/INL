matrix = parser_arff('data/vehicle.arff');
stdata = standarizer(matrix);

% -- K-Means for different K values
% ------------------------------------------------------

% Run the K-Means algorithm for K=2:5
execute_k_means(stdata, 2, 5);

% -- PCA
% ------------------------------------------------------

% Run the PCA alhorithm for an eigenvalue threshold of 1
[dataAfterPCA transformedData eVectors eValues mostInfFeatures] = pca(stdata,1);