function [ kPredictors ] = weightedKNN(TrainMatrix, current_instance, K, r)
%kNN Summary of this function goes here
%   Detailed explanation goes here

    %Get the eigenValues and the rank of the attributes according to its
    %importance
    [ ~, ~, ~, eValues, informativeFeatures] = pca(TrainMatrix(:,1:end-1), 0);
    
    %Calc the sum of the eigenValues
    sumEValues = sum(eValues);
    
    weightedMatrix = TrainMatrix;
    
    [~, numAttributes] = size(informativeFeatures);
    
    %Generates a weightedVersion of the TrainMatrix accordint to the
    %eigenValue associated to each attribute
    for i = 1:numAttributes
        weightedMatrix(:,informativeFeatures(i)) = weightedMatrix(:,informativeFeatures(i))*(eValues(i)/sumEValues);
    end

    %Calc the minkowski distance from the current_instance to the predictors
    distances = pdist2(weightedMatrix(:,1:end-1),current_instance,'minkowski',r);

    %Calc the minimum distances
    [~, kPindexs] = min(distances, [], 2);

    %Return a list of K predictors
    kPredictors = TrainMatrix(kPindexs >= K, :);
end

