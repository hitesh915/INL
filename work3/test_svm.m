function [ predicted, f ] = test_svm( test, svmStruct )
%TEST_SVM Summary of this function goes here
%   Detailed explanation goes here

    % Get hyperplane parameters
    w = svmStruct.w;
    b = svmStruct.b;
    
    % Standarize test data
    test = standarizer(test, svmStruct.meanTrain, svmStruct.stdTrain);
    
    % Initialize f vector
    f = zeros(size(test,1),1);
    
    % Classify using the linear kernel
    if strcmp(svmStruct.kernel,'linear')
        f = zeros(size(test,1),1);
        for i = 1:size(test,1)
            f(i) = w'*test(i,:)' + b;
        end
        predicted = sign(f);
        
    % Classify using the RBF kernel
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