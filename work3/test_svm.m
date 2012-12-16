function [ predicted, f ] = test_svm( svmStruct, test )
%TEST_SVM Summary of this function goes here
%   Detailed explanation goes here

    w = svmStruct.w;
    b = svmStruct.b;
    
    % Standarize test data
    test = bsxfun(@minus, test, svmStruct.meanTrain);
    test = bsxfun(@rdivide, test, svmStruct.stdTrain);
    test(isnan(test)) = 0;
    
    % Initialize f vector
    f = zeros(size(test,1),1);
    
%     w = [0, 0];
%     for i = 1:size(svmStruct.w,1)
%         w = w + svmStruct.sv_alphas(i)*svmStruct.sv_labels(i)*svmStruct.sv_points(i,:);
%     end
    
    if strcmp(svmStruct.kernel,'linear')
        f = zeros(size(test,1),1);
        for i = 1:size(test,1)
            f(i) = svmStruct.w'*test(i,:)'+b;
        end
        predicted = sign(f);
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