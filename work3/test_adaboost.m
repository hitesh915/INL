function [labels] = test_adaboost(model, data)
    function y = classify_data(h, x)
        if(h.direction == 1)
            y =  double(x(:,h.dimension) >= h.threshold);
        else
            y =  double(x(:,h.dimension) < h.threshold);
        end
        y(y == 0) = -1;
    end

    % Retrieve weak models
    models = model.models;
    
    % Standardize test data
    data = bsxfun(@minus, data, model.mean);
    data = bsxfun(@rdivide, data, model.std);
    data(isnan(data)) = 0;
    
    % Add results of the single weak classifiers weighted by their alpha 
    labels=zeros(size(data,1),1);
    for t=1:length(model);
        labels = labels + models(t).alpha * classify_data(models(t), data);
    end
    
    % If sum of weak classifiers < 0, class = -1, +1 otherwise
    labels = sign(labels);
end

