function [model] = train_adaboost(labels, data, T, weakClassifier)
%TRAIN_ADABOOST train a adaboost algorithm
%   INPUT:
%       labe
    
    %Assign a function handler that acts as a weak classifier
    if nargin < 4
        weakClassifier = 'svm';
    end
    
    switch weakClassifier
        case 'svm'
            weakC = @train_svm;
            weakTester =  @test_svm;
        case 'perceptron'
            weakC = @train_Perceptron;
            weakTester =  @test_Perceptron;
        case 'sigmoidalPerceptron'
            weakC = @train_PerceptronSigmoid;
            weakTester =  @test_PerceptronSigmoid;
    end
    
    %Obtain the data relevant to the adaboost t evaluation: labels
    %predicteds, error of the prediction, and the model
    function [predicted, error, model] = train_weak_classifier(data,labels,weights)
        %Sample the data using the weights
        [sample, labelsSample] = sus(data,labels,weights);
        
        %Get the model
        model = feval(weakC,labelsSample,sample);
        
        %Test the model
        predicted = feval(weakTester,data,model);
        
        %Calculate the error of the model
        error = sum(predicted~=labels)/size(labels,1);
    end
    
    % Standarize data
    mean = nanmean(data);
    std = nanstd(data);
    data = standarizer(data);

    % Get size of training data
    m = size(data, 1);
    
    % Initialize perceptrons matrix
    models = struct;
    
    % Initialize vector of weights
    D = zeros(m,1);
    D(:) = 1/m;
    
    for ii=1:T
        % Train new classifier
        [results, error, w] = train_weak_classifier(data, labels, D);
        
        % If error >= 1/2, stop algorithm
        if error >= 0.5
            break
        end
        
        % Calculate alpha
        alpha = 1/2 * log((1 - error) / (max(error, eps)));
        
        % Store the model parameters
        models(ii).alpha = alpha;
        models(ii).w = w;

        % Update vector of weights
        D = D .* exp(-models(ii).alpha .* labels .* results);
        D = D ./ sum(D);
    end
    
    % Prepare response model
    model = struct;
    model.models = models;
    model.t = T;
    model.mean = mean;
    model.std = std;
    model.weakTester = weakTester;
end

