function [ result ] = rbfKernel( data1, data2, sigma )
%RBFFUNCTION Summary of this function goes here
%   Detailed explanation goes here

    result = zeros(size(data1,1), size(data2,1));
    for dI = 1:size(data1,1)
        for dJ = 1:size(data2,1)
            result(dI, dJ) = exp(-(norm(data1(dI,:)-data2(dJ,:))^2)/(2*(sigma^2)));
        end
    end
end