function [ predicted ] = test_svm( svmStruct, test )
%TEST_SVM Summary of this function goes here
%   Detailed explanation goes here

    w = svmStruct.w;
    b = svmStruct.b;
    test = standarizer(test);
    
    
    if strcmp(svmStruct.kernel,'linear')
        predicted = sign(test*w+b);
    else
%         kernel = trainData*trainData' / sigma^2;
%         d = diag(kernel);
%         kernel = kernel - ones(n,1)*d'/2;
%         kernel = kernel - d*ones(1,n)/2;
%         kernel = exp(kernel);
%         
%         predicted = sign(kernel*((svmStruct.alpha).*svmStruct.labels)+b);

        predicted = zeros(size(test,1),1);
        for n = 1:size(test,1)
            kernels = rbfKernel(svmStruct.sv_points,test(n,:), svmStruct.sigma);
            result = 0;
            for i = 1:size(svmStruct.sv_points,1)
                result = result + (svmStruct.sv_alphas(i) * svmStruct.sv_labels(i) * kernels(i) + b);
            end
            predicted(n) = sign(result);
        end
    end
end