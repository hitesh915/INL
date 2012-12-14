function [ predicted ] = test_svm( svmStruct, test )
%TEST_SVM Summary of this function goes here
%   Detailed explanation goes here

    w = svmStruct.w
    b = svmStruct.b
    
    if svmStruct.kernel == 'linear'
        predicted = sign(test*w+b);
    else
        
    end


end

