function [labels] = test_adaboost(model, data)
    % Standarize test data
    data = bsxfun(@minus, data, model.meanTrain);
    data = bsxfun(@rdivide, data, model.stdTrain);
    data(isnan(data)) = 0;

    % Get number of individuals
    nind = size(data,1);
    
    labels = zeros(size(data,1),1);
    
    for i = 1:nind
        for t = 1:model.T
            output = sign(sum(data(i,:)' .* model.models(t)));
            labels(i) = labels(i) + model.alphas(t) * output;
        end
        
        labels(i) = sign(labels(i));
    end
end

