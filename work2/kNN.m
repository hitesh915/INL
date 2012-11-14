function [ kPredictors ] = kNN(TrainMatrix, current_instance, K, r)
%kNN Summary of this function goes here
%   Detailed explanation goes here

    %Calc the minkowski distance from the current_instance to the predictors
    distances = pdist2(TrainMatrix(:,1:end-1),current_instance,'minkowski',r);

    %Calc the minimum distances
    [~, kPindexs] = min(distances, [], 2);

    %Return a list of K predictors
    kPredictors = TrainMatrix(kPindexs >= K, :); 

end

