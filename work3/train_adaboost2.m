function [model] = train_adaboost(labels, data, T)
%TRAIN_SVM Summary of this function goes here
%   Detailed explanation goes here

    % Standarize data
    mean = nanmean(data);
    std = nanstd(data);
    data = standarizer(data);

%     % Function to train a perceptron classifier
%     function [weights offset] = weak_learn(individuals, lbls, rates)
%         [nindividuals nnodes] = size(individuals,1);
%         
%         weights = zeros(nnodes+1, 1);
%         offset = 1;
%         
%         for it = 1:200
%             for ind = 1:nindividuals
%                 individual = individuals(ind,:)';
%                 out = sign(sum([offset;individual] .* weights));
%                 %weights = weights + rates(ind) * (lbls(ind) - out) * individual;
%                 if lbls(ind) ~= out
%                     weithts(1) = weight +
%                     weights(2:end,1) = weights(2:end,1) + rates(ind) * lbls(ind) * individual;
%                 end
%             end
%         end
%     end

    function [weights offset] = weak_learn(individuals, lbls, rates)
        weights = zeros(size(individuals,1),size(individuals,2)+1);
        bias = ones(size(individuals,1),1);
        individuals = [bias, individuals];
        for it = 1:100
            for n = 1:size(individuals, 1)
                out = sign(individuals(n,:)*weights(n)');
                if out ~= lbls(n)
                    weights(n,:) = (weights(n,:)+lbls(n)*rates(n)*individuals(n));
                end
            end
        end        
    end

%     function [w] = perceptron(X,Y,w_init,D)
%         w = w_init;
%         X = X';
%         for iteration = 1 : 100  %<- in practice, use some stopping criterion!
%           for ii = 1 : size(X,2)         %cycle through training set
%             if sign(w'*X(:,ii)) ~= Y(ii) %wrong decision?
%               w = w + X(:,ii) * Y(ii);   %then add (or subtract) this point to w
%             end
%           end
%           w
%           X
%           sum(sign(w*X')~=Y)/size(X,2)   %show misclassification rate
%         end
%     end

    % Function to test the perceptron classifier
    function [labels] = weak_classifier(individuals, weights, offset)
        nindividuals = size(individuals,1);
        
        labels = zeros(nindividuals,1);
        for ind = 1:nindividuals
            labels(ind) = sign(sum(individuals(ind,:)' .* weights) + offset);
        end
    end

    % Get size of training data
    m = size(data, 1);
    
    % Initialize perceptrons matrix
    models = [];
    error = [];
    
    % Initialize vector of weights
    D = zeros(m,1);
    D(:) = 1/m;
    
    t = T;
    for i=1:T
        results = nan;
        weights = nan;
        offset = nan;
        
        size(data)
        
        weights = weak_learn(data, labels, D);
        
        bias = ones(size(data,1),1);
        
        size(weights)
        size(data)
        
        predicted = sign(sum([bias,data].*weights,2));
        
        sum(predicted~=labels)
        
        epsilon_t = sum(predicted~=labels)/size(data,1);
        epsilon_t
        error = [error;epsilon_t];
        
        % If epsilon >= 1/2, stop algorithm
        if epsilon_t >= 0.5 || epsilon_t == 0
            t = i;
            break
        end
        
        D(labels==predicted) = D(labels==predicted)*(epsilon_t/(1-epsilon_t));
        
        D = D / sum(D);
        
        % Save model of this iteration
        models = [models;weights];
    end
    
    % Prepare response model
    model = struct('errors',error,'T', t, 'models', models,'meanTrain', mean, 'stdTrain', std);
end

