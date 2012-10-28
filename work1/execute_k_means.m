function [ output_args ] = execute_k_means( matrix, minK, maxK )
    %EXECUTE_K_MEANS Execute K-Means for the specified range of K over 'matrix'
    %   This function executes the K-Means algorithm for all the values of K in
    %   the range minK:maxK, and shows an histogram for those values and it's
    %   inertia.

    for k = minK:maxK
        % Initialize best seed result value
        best_inertia = nan;
        best_output = nan;
        best_centroids = nan;

        % Calculate with different seeds and select best
        for seed = 1:3
            %Compute the k clusters with k-means
            [output, centroids, inertias] = k_means(matrix, k,seed);

            %Select this result if lower inertia than the previous ones
            best_inertia = min(best_inertia, sum(inertias));
            if best_inertia == sum(inertias)
                best_output = output;
                best_centroids = centroids;
            end
        end
        best_inertia

        %Use the MDS algorithm to reduce the dimensionality
        dimReduced = mds(matrix,3);

        %Show a scatter plot in 3D of the clustered data with the best seed
        subplot(2,2,k-1);
        scatter3(dimReduced(:,1),dimReduced(:,2),dimReduced(:,3), 20, best_output, 'filled');
        title(strcat('k=',int2str(k)));
    end
end

