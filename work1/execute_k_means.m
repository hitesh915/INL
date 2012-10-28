function [ output_args ] = execute_k_means( matrix, minK, maxK, usePCA)
    %EXECUTE_K_MEANS Execute K-Means for the specified range of K over 'matrix'
    %   This function executes the K-Means algorithm for all the values of K in
    %   the range minK:maxK, and shows an histogram for those values and it's
    %   inertia.

    

    % Fill in unset optional values.
    switch nargin
        case 3
            usePCA = false;
    end
    
    if usePCA
        [dataAfterPCA transformedData eVectors eValues mostInfFeatures] = pca(matrix);
        mostInfFeatures
    end
    
    for k = minK:maxK
        
        if usePCA
            [best_output, best_centroids, best_inertia] = k_means(transformedData', k,1,50);
        else
            [best_output, best_centroids, best_inertia] = k_means(matrix, k,1,50);
        end

        %Use the MDS algorithm to reduce the dimensionality
        dimReduced = mds(matrix,3);

        %Show a scatter plot in 3D of the clustered data with the best seed
        subplot(2,2,k-1);
        scatter3(dimReduced(:,1),dimReduced(:,2),dimReduced(:,3), 20, best_output, 'filled');
        title(strcat('k=',int2str(k)));
    end
end

