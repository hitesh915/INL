function [ predicted ] = test_Perceptron( test,w )
%TEST_PERCEPTRON Summary of this function goes here
%   Detailed explanation goes here
    predicted = zeros(size(test,1),1);
    offset = -1;
    for p = 1:size(test,1)
        predicted(p) = [test(p,:),offset]*w;
        if predicted(p) >= 0
            predicted(p) = 1;
        else
            predicted(p) = 0;
        end
    end
    predicted = predicted - 0.5;
    predicted = sign(predicted);
end

