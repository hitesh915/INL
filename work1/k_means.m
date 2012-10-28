function [ best_output, best_centroids, best_inertia ] = k_means( dataMatrix, k, seed, repetitions )
    %KMEANS Applies the K-Means algorithm to a given data set
    %   For a data set represented with a matrix of column features and row
    %   individuals, a K-Means algorithm is applied to generate k clusters.
    %
    %   INPUTS:
    %     - dataMatrix = A matrix with column features and row individuals
    %     - k          = Number of clusters to calculate
    %     - seed       = Seed for a random generator of individuals cluster
    %     - repetitions= Number of repetitions of the algorithm, returns
    %     the best clusters so far
    %
    %   OUTPUTS:
    %     - best_output     = Vector with the cluster for each individual
    %     - best_centroids  = Centroids of the clusters (1 row per cluster)
    %     - best_inertia    = Vector with the inertia of each cluster

    % Fill in unset optional values.
    switch nargin
        case 2
            seed = 1;
            repetitions = 1;
        case 3
            repetitions = 1;
    end
    
    function [centroids] = getCentroids(dataMatrix,membership)
        centroids = zeros(k,nCol, 'double');
        for j = 1:k
            centroidK = mean(dataMatrix(membership == j,:));
            centroids(j,:) = centroidK;
        end
    end

    % Initialize best seed result value
    best_inertia = nan;
    best_output = nan;
    best_centroids = nan;

    % Calculate with different seeds and select best
    for s = seed:repetitions+seed

        [nRows nCol] = size(dataMatrix);

        %Init. random generator
        rng(s);

        %vector of membership generated randomly
        membership = randi([1,k],nRows,1);

        %get the initial centroids
        centroids = -3 + (3+3).*rand(k,nCol);

        convergence = false;


        while ~convergence
            distances = pdist2(dataMatrix,centroids).^2;
            [~, membership] = min(distances,[],2);
            newCentroids = getCentroids(dataMatrix,membership);
            if centroids == newCentroids
                convergence = true;
            else
                centroids = newCentroids;

                %Check if there are any centroid without members, if that
                %centroid exists, generate a new one randomly
                nans = isnan(centroids(:,1));
                if max(nans)
                    for i = find(nans)'
                        centroids(i,:) = -3 + (3+3).*rand(1,nCol);
                    end
                end
            end
        end

        sumWD = zeros(k,1,'double');
        for i = 1:k
            wd = sum(pdist2(dataMatrix(membership == i,:), centroids(i,:)).^2);
            sumWD(i,:) = wd;
        end
        
        %Select this result if lower inertia than the previous ones
        best_inertia = min(best_inertia, sum(sumWD));
        if best_inertia == sum(sumWD)
            best_output = membership;
            best_centroids = centroids;
        end
    end
end

