function [ w ] = train_Perceptron(labels,data)
%TRAIN_PERCEPTRON Summary of this function goes here
%   Detailed explanation goes here

    w = rand(size(data,2)+1,1);
    offset = -1;
    learnRate = 0.2;
    labels(labels == -1) = 0;

    for iterations = 1:100
        for p = 1:size(data,1)
            out = [data(p,:),offset]*w;
            if out >= 0
                out = 1;
            else
                out = 0;
            end
            if out ~= labels(p)
                for i = 1:size(data,2)
                    w(i) = w(i)+2*learnRate*(labels(p)-out)*data(p,i);
                end
                w(end) = w(end)+2*learnRate*(labels(p)-out)*offset;
            end
        end
    end
    
end

