function [model] = train_adaboost(labels, data, T)
%TRAIN_SVM Summary of this function goes here
%   Detailed explanation goes here

    % Standarize data
    mean = nanmean(data);
    std = nanstd(data);
    data = standarizer(data);

    % Function to train a perceptron classifier
    function [weights] = weak_learn(individuals, lbls, rates)
        [nindividuals nnodes] = size(individuals);
        
        weights = zeros(nnodes, 1);
        
        for it = 1:200
            for ind = 1:nindividuals
                individual = individuals(ind,:)';
                out = sign(sum(individual .* weights));
                weights = weights + rates(ind) * (lbls(ind) - out) * individual;
                if sum(isnan(weights)) > 1
                    break;
                end
            end
        end
    end

    % Function to test the perceptron classifier
    function [labels] = weak_classifier(individuals, weights)
        nindividuals = size(individuals,1);
        
        labels = zeros(nindividuals,1);
        for ind = 1:nindividuals
            labels(ind) = sign(sum(individuals(ind,:)' .* weights));
        end
    end

    % Function to test perceptron classifier

    % Get size of training data
    m = size(data, 1);
    
    % Initialize perceptrons matrix
    models = zeros(T,size(data,2));
    alphas = zeros(T,1);
    
    % Initialize vector of weights
    D = zeros(m,1);
    D(:) = 1/m;
    
    for i=1:T
        
        % Train perceptron classifier
        weights = weak_learn(data, labels, D);
        results = weak_classifier(data, weights);
        
        % Calculate epsilon
        epsilon_t = sum(D .* (results ~= labels));
        
        % Choose alpha-t
        alpha_t = 1/m * log((1-epsilon_t) / epsilon_t);

        % Update vector of weights
        for j=1:m
            D = D .* exp(-alpha_t * (labels .* results));
            D = D / sum(D);
        end
        
        % Save model of this iteration
        models(i,:) = weights;
        alphas(i) = alpha_t;
    end
    
    % Prepare response model
    model = struct('T', T, 'models', models, 'alphas', alphas, 'meanTrain', mean, 'stdTrain', std);
end

