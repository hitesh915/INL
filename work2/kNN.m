function [ kPredictors ] = kNN(TrainMatrix, current_instance, K, r)
%kNN Summary of this function goes here
%   Detailed explanation goes here

%Calc the minkowski distance from the current_instance to the predictors
    distances = pdist2(TrainMatrix(:,1:end-1),current_instance,'minkowski',r);
 
    [~, distIndex] = sort(distances);
    [~, distRank] = sort(distIndex);
    
    kPredictors = TrainMatrix(distRank <= K,:);

end