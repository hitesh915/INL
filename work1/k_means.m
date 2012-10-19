function [ membership, centroids, sumWD ] = k_means( dataMatrix, k, seed )
%KMEANS Summary of this function goes here
%   Detailed explanation goes here

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

