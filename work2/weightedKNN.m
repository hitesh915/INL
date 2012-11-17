function [ kPredictors, kWeightedDistances ] = weightedKNN(TrainMatrix, current_instance, K, r, weights)
%kNN Summary of this function goes here
%   Detailed explanation goes here

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