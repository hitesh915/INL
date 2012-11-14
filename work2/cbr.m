function [ accuracy ] = cbr( trainMatrix, testMatrix, K, r )
    %CBR Summary of this function goes here
    %   Detailed explanation goes here

    % Get number of instances on the test matrix
    tmSize = size(testMatrix);
    tmSize = tmSize(1);
    
    % Initialize classification success matrix
    clsSuccess = [];
    
    % Classify the test individuals
    for i=0:tmSize
        % Get instance and predictors
        instance = testMatrix(i,:);
        predictors = kNN(trainMatrix, instance, K, r);
        
        % Generate counter vector for the classes
        numClasses = zeros(1,max(predictors(:,end)));
        
        % 
    end
end