function [ predicted, f ] = test_svm( svmStruct, test )
%TEST_SVM Summary of this function goes here
%   Detailed explanation goes here

    w = svmStruct.w;
    b = svmStruct.b;
    
    % Standarize test data
    test = bsxfun(@minus, test, svmStruct.meanTrain);
    test = bsxfun(@rdivide, test, svmStruct.meanTrain);
    test(isnan(test)) = 0;
    
    % Initialize f vector
    f = zeros(size(test,1),1);
    
    if strcmp(svmStruct.kernel,'linear')
        predicted = sign(test*w+b);
    else
        kernels = rbfKernel(svmStruct.sv_points,test, svmStruct.sigma);

        predicted = zeros(size(test,1),1);
        for n = 1:size(test,1)
            result = 0;
            for i = 1:size(svmStruct.sv_points,1)
                result = result + (svmStruct.sv_alphas(i) * svmStruct.sv_labels(i) * kernels(i,n));
            end
            f(n) = result + b;
            predicted(n) = sign(f(n));
        end
    end
end