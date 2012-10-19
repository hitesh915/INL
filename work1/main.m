matrix = parser_arff('data/vehicle.arff');
matrix = standarizerS(matrix);
[membership,centroids, sumD] = k_means(matrix,3,42);