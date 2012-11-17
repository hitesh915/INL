function [ kPredictors, kWeightedDistances ] = weightedKNN(TrainMatrix, current_instance, K, r, weights)
%kNN Summary of this function goes here
%   Detailed explanation goes here

    distances = zeros(1,size(TrainMatrix,1));
    for i = 1:size(TrainMatrix,1)
        dif = TrainMatrix(i,1:end-1) - current_instance;
        pow = dif.^r;
        pow = abs(pow);
        weightPow = pow.*abs(weights);
        distances(i) = sum(weightPow)^(1/r);
    end

    %calc the k predictors
    [~, distIndex] = sort(distances);
    [~, distRank] = sort(distIndex);
    
    iPredictors = distRank <= K;
    kPredictors = TrainMatrix(iPredictors,:);
    kWeightedDistances = distances(iPredictors);
end