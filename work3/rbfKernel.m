function [ result ] = rbfKernel( data1, data2, sigma )
%RBFFUNCTION This function implements the computation of gaussian RBF
%kernel function.
%   INPUT:
%       -data1: dataset 1
%       -data2: dataset 2
%   OUTPUT:
%       -result: the kernel function result of the two input datasets.

    result = zeros(size(data1,1), size(data2,1));
    for dI = 1:size(data1,1)
        for dJ = 1:size(data2,1)
            result(dI, dJ) = exp(-(norm(data1(dI,:)-data2(dJ,:))^2)/(2*(sigma^2)));
        end
    end
end