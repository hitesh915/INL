function [model] = train_adaboost(labels, data, T)
%TRAIN_SVM Summary of this function goes here
%   Detailed explanation goes here

    % Standarize data
    mean = nanmean(data);
    std = nanstd(data);
    data = standarizer(data);

    % Function to train a perceptron classifier
    function [weights offset] = weak_learn(individuals, lbls, rates)
        [nindividuals nnodes] = size(individuals);
        
        weights = zeros(nnodes, 1);
        offset = 0;
        
        for it = 1:200
            for ind = 1:nindividuals
                individual = individuals(ind,:)';
                out = sign(sum(individual .* weights) + offset);
                %weights = weights + rates(ind) * (lbls(ind) - out) * individual;
                if lbls(ind) ~= out
                    weights = weights + rates(ind) * lbls(ind) * individual;
                    offset = offset + rates(ind) * lbls(ind);
                end
            end
        end
    end

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
    models = zeros(T,size(data,2));
    alphas = zeros(T,1);
    offsets = zeros(T,1);
    
    % Initialize vector of weights
    D = zeros(m,1);
    D(:) = 1/m;
    
    t = T;
    for i=1:T
        results = nan;
        weights = nan;
        offset = nan;
        
        epsilon_t = inf;
        for sm = 1:20
            % Sample dataset according to weights distribution
            n = m/1.5;
            indices = zeros(n,1);
            random_number = rand(1,n);
            for k =1:n
                location = find(random_number(k) <= cumsum(D), 1);
                if isempty(location)
                    indices(k)=1;
                else
                    indices(k)=location;
                end
            end
            sdata = data(indices,:);
            slabels = labels(indices,:);
            
            % Train perceptron classifier
            [tweights, toffset] = weak_learn(sdata, slabels, D);
            tresults = weak_classifier(data, tweights, toffset);

            % Calculate epsilon
            tepsilon_t = sum(D .* (tresults ~= labels));
            %tepsilon_t = 1/m * sum(tresults ~= labels);
            
            if tepsilon_t < epsilon_t
                epsilon_t = tepsilon_t;
                weights = tweights;
                offset = toffset;
                results = tresults;
            end
        end
        
        % If epsilon >= 1/2, stop algorithm
        if epsilon_t >= 0.5
            t = i;
            break
        end
        
        % Choose alpha-t (confidence value)
        %alpha_t = 1/2 * log((1-epsilon_t) / epsilon_t);
        alpha_t = -log(epsilon_t / (1 - epsilon_t));

        % Update vector of weights
        correct = find(labels == results);
        D(correct) = D(correct) * epsilon_t / (1 - epsilon_t);
        %D = D .* exp(-alpha_t * (labels .* results));
        D = D / sum(D);
        
        % Save model of this iteration
        models(i,:) = weights;
        alphas(i) = alpha_t;
        offsets(i) = offset;
    end
    
    % Prepare response model
    model = struct('T', t, 'models', models, 'alphas', alphas, 'meanTrain', mean, 'stdTrain', std, 'offsets', offsets);
end

