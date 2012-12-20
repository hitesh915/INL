function [ w ] = train_Perceptron(labels,data)
%TRAIN_PERCEPTRON train a perceptron with the given data
%
%   ACTIVATION FUNCTION: STEP
%   INPUT:
%       - labels: The labels of each instance of the training dataset
%       - data: the data used to train the dataset
%   OUTPUT:
%       - model: the model that contais all the weights needed to perform 
%       the classification.

    % Initializate the weights at random
    w = rand(size(data,2)+1,1);
    
    % Specify the offset
    offset = -1;
    
    % Specify the learning step
    learnRate = 0.2;
    
    %Set the -1 labels to 0
    labels(labels == -1) = 0;

    %Iterate 100 times in order to setup the weigts.
    for iterations = 1:100
        for p = 1:size(data,1)
            out = [data(p,:),offset]*w;
            if out >= 0
                out = 1;
            else
                out = 0;
            end
            
            %If a training example is missclassified update the weights
            if out ~= labels(p)
                %Weights update
                for i = 1:size(data,2)
                    w(i) = w(i)+2*learnRate*(labels(p)-out)*data(p,i);
                end
                w(end) = w(end)+2*learnRate*(labels(p)-out)*offset;
            end
        end
    end
    
end

