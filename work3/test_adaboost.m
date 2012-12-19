function [labels] = test_adaboost(data,model)
    % Retrieve weak models
    models = model.models;
    
    % Standardize test data
    data = standarizer(data, model.meanTrain, model.stdTrain);
    
    % Add results of the single weak classifiers weighted by their alpha 
    labels=zeros(size(data,1),1);
    for t=1:length(model);
        labels = labels + models(t).alpha * feval(model.weakTester, data,models(t).w);
    end
    
    % If sum of weak classifiers < 0, class = -1, +1 otherwise
    labels = sign(labels);
end

