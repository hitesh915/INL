matrix = parser_arff('data/breast-w.arff');
matrix1 = standarizerS(matrix);
matrix2 = standarizer(matrix);

%Compute the k clusters with k-means
[output, centroids, sum] = k_means(matrix1, 2, 7);

%Use the MDS algorithm to reduce the dimensionality
dimReduced = mds(matrix1,3);

%Show a scatter plot in 3D of the clustered data.
scatter3(dimReduced(:,1),dimReduced(:,2),dimReduced(:,3), 20, output, 'filled');

%Compute the PCA transform
%[dataAfterPCA eVectors eValues] = pca(matrix,1);
