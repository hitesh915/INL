function [ kPredictors ] = weightedKNN(TrainMatrix, current_instance, K, r)
%kNN Summary of this function goes here
%   Detailed explanation goes here

    %standarize data
    [stdAttributes, mean, stdDev] = standarizer(TrainMatrix(:,1:end-1));
    stdTrain = [stdAttributes, TrainMatrix(:,end)];
    stdCurrent = (current_instance-mean)./stdDev;
    stdCurrent(isnan(stdCurrent)) = 0;
    
    %Get the weights using FS filter relieff
    [~, weights] = relieff(stdTrain(:,1:end-1),stdTrain(:,end),K, 'method', 'classification', 'categoricalx', 'off');
    
    %ranging the weights
    weights = weights - min(weights);
    weights = weights./range(weights);
    
    distances = zeros(1,size(stdTrain,1));
    for i = 1:size(stdTrain,1)
        dif = stdTrain(i,1:end-1) - stdCurrent;
        pow = dif.^r;
        pow = abs(pow);
        weightPow = pow.*abs(weights);
        distances(i) = sum(weightPow)^(1/r);
    end

    %calc the k predictors
    [~, distIndex] = sort(distances);
    [~, distRank] = sort(distIndex);
    
    kPredictors = TrainMatrix(distRank <= K,:);
    
end