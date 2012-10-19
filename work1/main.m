matrix = parser_arff('data/breast-w.arff');
matrix = standarizerS(matrix);
[membership,centroids, sumD] = k_means(matrix,3,42);
[dataAfterPCA eVectors eValues] = pca(matrix,1);