function [ kPredictors, kDistances ] = kNN(TrainMatrix, current_instance, K, r)
    %kNN This funcion implements the K-nearest-neighbors algorithm,
    %  obtaining the K nearest predictors in the TrainMatrix for the
    %  current_instance.
    %
    %  INPUTS:
    %    TrainMatrix = Matrix containing the predictors to use
    %    current_instance = Instance with the nearest neighbors to the
    %    K = Number of neighbors to retrieve
    %    r = Minkowski value for the type of distance to use
    %
    %  OUTPUTS:
    %    kPredictors = Vector with the K nearest predictors from the
    %    TrainMatrix
    %    kDistances = Vector with the K distances to the nearest predictors

    % Calculate the minkowski distance from current_instance to predictors
    distances = pdist2(TrainMatrix(:,1:end-1),current_instance,'minkowski',r);
 
    % Sort the predictors by distance (ascending)
    [~, distIndex] = sort(distances);
    [~, distRank] = sort(distIndex);
    
    % Select the K nearest predictors and their distances
    iPredictors = distRank <= K;
    kPredictors = TrainMatrix(distRank <= K,:);
    kDistances = distances(iPredictors);
end