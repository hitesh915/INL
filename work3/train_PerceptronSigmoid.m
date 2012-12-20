function [ w ] = train_PerceptronSigmoid( labels,data)
%TRAIN_PERCEPTRONSIGMOID train a perceptron with the given data
%
%   ACTIVATION FUNCTION: Logistic
%   INPUT:
%       - labels: The labels of each instance of the training dataset
%       - data: the data used to train the dataset
%   OUTPUT:
%       - model: the model that contais all the weights needed to perform 
%       the classification.
    
    %Initializate the weights at random
    w = rand(size(data,2)+1,1);
    
    %Set the value of the offset and the learning rate
    offset = -1;
    learnRate = 0.2;
    
    %Set the -1 labels to 0
    labels(labels == -1) = 0;

    %Iterate 100 times in order to setup the weigts.
    for iterations = 1:100
        for p = 1:size(data,1)
            out = 1/(1+exp(-([data(p,:),offset]*w)));
            
            %Update the weights
            for i = 1:size(data,2)
                w(i) = w(i)+2*learnRate*(labels(p)-out)*out*(1-out)*data(p,i);
            end
            w(end) = w(end)+2*learnRate*(labels(p)-out)*out*(1-out)*offset;
        end
    end
end