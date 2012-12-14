function [ predicted ] = test_svm( svmStruct, test )
%TEST_SVM Summary of this function goes here
%   Detailed explanation goes here

    w = svmStruct.w;
    b = svmStruct.b;
    test = standarizer(test);
    
    
    if strcmp(svmStruct.kernel,'linear')
        predicted = sign(test*w+b);
    else
        kernel = trainData*trainData' / sigma^2;
        d = diag(kernel);
        kernel = kernel - ones(n,1)*d'/2;
        kernel = kernel - d*ones(1,n)/2;
        kernel = exp(kernel);
        
        predicted = sign(kernel*((svmStruct.alpha).*svmStruct.labels)+b);
    end
end