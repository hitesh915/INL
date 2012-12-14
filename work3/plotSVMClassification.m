function [ ] = plotSVMClassification(data, class, w, b, svii)
%PLOTSVMCLASSIFICATION Summary of this function goes here
%   Detailed explanation goes here
    x = -0.6:0.001:0.6;
    
    yPart = (b - x * w(1))/w(2);
    
    data1 = data(class == 1,:);
    dataM1 = data(class == -1,:);
    
    %ToDo add a input parameter that give access to the supportVectors
    %index
    supportVectors = data(svii, :);
    
    hold on
    
    scatter(data1(:,1), data1(:,2), 'b.');
    scatter(dataM1(:,1), dataM1(:,2),'rx');
    scatter(supportVectors(:,1), supportVectors(:,2), 'go');
    
    plot(x',yPart')

end

