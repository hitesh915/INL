function [ kPredictors ] = kNN(TrainMatrix, current_instance, K, r)
%kNN Summary of this function goes here
%   Detailed explanation goes here

    %standarize data
    [stdAttributes, mean, stdDev] = standarizer(TrainMatrix(:,1:end-1));
    stdTrain = [stdAttributes, TrainMatrix(:,end)];
    stdCurrent = current_instance-mean;
    stdCurrent = stdCurrent./stdDev;
    stdCurrent(isnan(stdCurrent)) = 0;


    %Calc the minkowski distance from the current_instance to the predictors
    distances = pdist2(stdTrain(:,1:end-1),stdCurrent,'minkowski',r);
 
    [~, distIndex] = sort(distances);
    [~, distRank] = sort(distIndex);
    
    kPredictors = TrainMatrix(distRank <= K,:);

end