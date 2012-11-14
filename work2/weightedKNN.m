function [ kPredictors ] = weightedKNN(TrainMatrix, current_instance, K, r)
%kNN Summary of this function goes here
%   Detailed explanation goes here

    %Get the eigenValues and the rank of the attributes according to its
    %importance
    [ ~, ~, ~, eValues, informativeFeatures] = pca(TrainMatrix(:,1:end-1), 0);
    
    [~, numAttributes] = size(informativeFeatures);
    
    %Order the eValues depending on the index of its attribute
    orderedEigenValues = zeros(numAttributes);
    for i = 1:numAttributes
        orderedEigenValues(informativeFeatures(i)) = eValues(i);
    end
    
    %ScaleEigenvalues
    scaleEigenValues = orderedEigenValues./sum(eValues);
        
    
    %Calc the minkowski distance from the current_instance to the predictors
    distances = zeros(1,size(TrainMatrix,1));
    for i = 1:size(TrainMatrix,1)
        dif = TrainMatrix(i,1:end-1) - current_instance;
        pow = dif.^r;
        weightPow = pow.*scaleEigenValues;
        distances(i) = sum(weightPow)^(1/r);
    end

    %Calc the minimum distances
    [~, kPindexs] = min(distances, [], 2);

    %Return a list of K predictors
    kPredictors = TrainMatrix(kPindexs >= K, :);
end