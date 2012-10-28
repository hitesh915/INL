function [ output_args ] = execute_algorithms( matrix, minK, maxK, useKM, usePCA)
    %EXECUTE_K_MEANS Execute K-Means for the specified range of K over 'matrix'
    %   This function executes the K-Means algorithm for all the values of K in
    %   the range minK:maxK, and shows an histogram for those values and it's
    %   inertia.


    % Fill in unset optional parameters.
    switch nargin
        case 3 % If algorithms not defined, use only K-Means
            useKM = true;
            usePCA = false;
        case 4 % If PCA not defined, don't use it
            usePCA = false;
    end
    
    
    % Run PCA if required
    if usePCA
        [dataAfterPCA transformedData eVectors eValues mostInfFeatures] = pca(matrix);
        mostInfFeatures
    end
    
    % Run K-Means if required
    if useKM
        % Select data set to use (depending on the PCA usage)
        if(usePCA)
            used_data = transformedData';
        else
            used_data = matrix;
        end
        
        % For different values of K (minK:maxK range)
        for k = minK:maxK
            % Apply K-Means algorithm
            [best_output, best_centroids, best_inertia] = k_means(used_data, k,1,50);
            
            % Show best inertia for the K value
            str = strcat('For k=',int2str(k));
            if usePCA, str=strcat(str, ' (using PCA)'); end
            strcat(str, ', best inertia=', num2str(best_inertia))

            %Use the MDS algorithm to reduce the dimensionality
            dimReduced = mds(matrix,3);

            %Show a scatter plot in 3D of the clustered data with the best seed
            subplot(2,2,k-1);
            scatter3(dimReduced(:,1),dimReduced(:,2),dimReduced(:,3), 20, best_output, 'filled');
            title(strcat('k=',int2str(k)));
        end
    end
end

