function [ predicted ] = test_svm( svmStruct, test )
%TEST_SVM Summary of this function goes here
%   Detailed explanation goes here

    w = svmStruct.w;
    b = svmStruct.b;
    test = standarizer(test);
    
    
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
            predicted(n) = sign(result + b);
        end
    end
end