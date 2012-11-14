function [ kPredictors ] = weightedKNN(TrainMatrix, current_instance, K, r)
%kNN Summary of this function goes here
%   Detailed explanation goes here

    %standarize data
    [stdAttributes, mean, stdDev] = standarizer(TrainMatrix(:,1:end-1));
    stdTrain = [stdAttributes, TrainMatrix(:,end)];
    stdCurrent = (current_instance-mean)./stdDev;

    %Get the eigenValues and the rank of the attributes according to its
    %importance
    [ ~, ~, ~, eValues, informativeFeatures] = pca(stdTrain(:,1:end-1), 0);
    
    [~, numAttributes] = size(informativeFeatures);
    
    %Order the eValues depending on the index of its attribute
    orderedEigenValues = zeros(1, numAttributes);
    for i = 1:numAttributes
        orderedEigenValues(informativeFeatures(i)) = eValues(i);
    end
    
    %ScaleEigenvalues
    scaleEigenValues = orderedEigenValues./sum(eValues);            
    
    %Calc the minkowski distance from the current_instance to the predictors
    distances = zeros(1,size(stdTrain,1));
    for i = 1:size(stdTrain,1)
        dif = stdTrain(i,1:end-1) - stdCurrent;
        pow = dif.^r;
        weightPow = pow.*scaleEigenValues;
        distances(i) = sum(weightPow)^(1/r);
    end

    %calc the k predictors
    [~, distIndex] = sort(distances);
    [~, distRank] = sort(distIndex);
    
    kPredictors = TrainMatrix(distRank <= K,:);
    
end