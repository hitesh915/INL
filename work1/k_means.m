function [ membership, centroids, sumWD ] = k_means( dataMatrix, k, seed )
    %KMEANS Applies the K-Means algorithm to a given data set
    %   For a data set represented with a matrix of column features and row
    %   individuals, a K-Means algorithm is applied to generate k clusters.
    %
    %   INPUTS:
    %     - dataMatrix = A matrix with column features and row individuals
    %     - k          = Number of clusters to calculate
    %     - seed       = Seed for a random generator of individuals cluster
    %
    %   OUTPUTS:
    %     - membership = Vector with the cluster for each individual
    %     - centroids  = Centroids of the clusters (1 row per cluster)
    %     - sumWD      = Vector with the inertia of each cluster

    function [centroids] = getCentroids(dataMatrix,membership)
        centroids = zeros(k,nCol, 'double');
        for j = 1:k
            centroidK = mean(dataMatrix(membership == j,:));
            centroids(j,:) = centroidK;
        end
    end

    [nRows nCol] = size(dataMatrix);
    
    %Init. random generator
    rng(seed);
    
    %vector of membership generated randomly
    membership = randi([1,k],nRows,1);
    
    %get the initial centroids
    centroids = getCentroids(dataMatrix, membership);
    
    convergence = false;
     
    while ~convergence
        distances = pdist2(dataMatrix,centroids);
        [~, membership] = min(distances,[],2);
        newCentroids = getCentroids(dataMatrix,membership);
        if centroids == newCentroids
            convergence = true;
        else
            centroids = newCentroids;
        end
    end
    
    sumWD = zeros(k,1,'double');
    for i = 1:k
        wd = sum(pdist2(dataMatrix(membership == i,:), centroids(i,:)));
        sumWD(i,:) = wd;
    end
end

