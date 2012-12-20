function [ result ] = linearKernel( data1, data2 )
%RBFFUNCTION This function implements the computation of linar kernel,
%basicaly a dot product.
%   INPUT:
%       -data1: dataset 1
%       -data2: dataset 2
%   OUTPUT:
%       -result: the dot product between the two input data specified on
%       the input

    %Initializate the matrix of results
    result = zeros(size(data1,1), size(data2,1));
    
    %Calculate the kernel function
    for dI = 1:size(data1,1)
        for dJ = 1:size(data2,1)
            result(dI, dJ) = data1(dI,:)*data2(dJ,:)';
        end
    end
end