function [ kPredictors, kWeightedDistances ] = weightedKNN(TrainMatrix, current_instance, K, r, weights)
    %kNN Finds the K nearest predictors for the current instance by using
    %  the r-th minkowski distance, and modifying the distance of each
    %  dimension according to its weight.
    %
    %  INPUTS:
    %    TrainMatrix = Matrix with all the predictors
    %    current_instance = Instance to classify
    %    K = Number of neighbors to retrieve
    %    r = Type of minkowski distance to use
    %    weights = Vector with the weight of each dimension
    %
    %  OUTPUTS:
    %    kPredictors = Vector with the K nearest predictors
    %    kWeightedDistances = Vector with the weighted distances of the K
    %    returned predictors

    %Calc the minkowski distance from the current_instance to the
    %predictors taking into account the weights of each attribute.
    distances = zeros(1,size(TrainMatrix,1));
    for i = 1:size(TrainMatrix,1)
        dif = TrainMatrix(i,1:end-1) - current_instance;
        pow = dif.^r;
        pow = abs(pow);
        weightPow = pow.*weights;
        distances(i) = sum(weightPow)^(1/r);
    end

    %calc the k predictors
    [~, distIndex] = sort(distances);
    [~, distRank] = sort(distIndex);
    
    %return the K predictors and the distance to the current_instance
    iPredictors = distRank <= K;
    kPredictors = TrainMatrix(iPredictors,:);
    kWeightedDistances = distances(iPredictors);
end