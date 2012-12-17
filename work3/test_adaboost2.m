function [labels] = test_adaboost(model, data)

    function [labels] = weak_classifier(individuals, weights, offset)
        nindividuals = size(individuals,1);
        
        labels = zeros(nindividuals,1);
        for ind = 1:nindividuals
            labels(ind) = sign(sum(individuals(ind,:)' .* weights) + offset);
        end
    end

    % Standarize test data
    data = bsxfun(@minus, data, model.meanTrain);
    data = bsxfun(@rdivide, data, model.stdTrain);
    data(isnan(data)) = 0;
    
    numberOfModels = size(model.models,1);
    
    modelWeight = zeros(numberOfModels,1);
    
    for i = 1:numberOfModels
        modelWeight(i) = -log(model.errors(i)/(1-model.errors(i)));
        test
        predicted = ;        
    end
    
end

