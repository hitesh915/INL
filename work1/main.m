matrix = parser_arff('data/vehicle.arff');
matrix1 = standarizerS(matrix);
matrix2 = standarizer(matrix);

[dataAfterPCA transformedData eVectors eValues mostInfFeatures] = pca(matrix1,1);

% -- K-Means for different K values
% ------------------------------------------------------

for k = 2:5
    % Initialize best seed result value
    best_inertia = nan;
    best_output = nan;
    best_centroids = nan;
    
    % Calculate with different seeds and select best
    for seed = 1:3
        %Compute the k clusters with k-means
        [output, centroids, inertias] = k_means(matrix1, k,seed);
    
        %Select this result if lower inertia than the previous ones
        best_inertia = min(best_inertia, sum(inertias));
        if best_inertia == sum(inertias)
            best_output = output;
            best_centroids = centroids;
        end
    end
        
    %Use the MDS algorithm to reduce the dimensionality
    dimReduced = mds(matrix1,3);

    %Show a scatter plot in 3D of the clustered data with the best seed
    subplot(2,2,k-1);
    scatter3(dimReduced(:,1),dimReduced(:,2),dimReduced(:,3), 20, best_output, 'filled');
    title(strcat('k=',int2str(k)));
end

% -- PCA
% ------------------------------------------------------

%Compute the PCA transform
[dataAfterPCA transformedData eVectors eValues featuredVectors] = pca(matrix1,1);
[~, mostInfFeatures] = max(featuredVectors);

