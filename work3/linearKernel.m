function [ result ] = linearKernel( data1, data2 )
%RBFFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    result = zeros(size(data1,1), size(data2,1));
    for dI = 1:size(data1,1)
        for dJ = 1:size(data2,1)
            result(dI, dJ) = data1(dI,:)*data2(dJ,:)';
        end
    end
end