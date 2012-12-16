function [model] = train_svm(labels, data, C, sigma)
%TRAIN_SVM Summary of this function goes here
%   Detailed explanation goes here

    data = standarizer(data);
    
    % If C is undefined, set C to 2
    if nargin < 3
       C = 2;
    end
    
    % Count number of data points
    n = size(data, 1);
    
    % Define linear/RBF kernel
    if nargin < 4
        kernel = linearKernel(data, data);
    else
        kernel = rbfKernel(data, data, sigma);
%         kernel = data*data' / sigma^2;
%         d = diag(kernel);
%         kernel = kernel - ones(n,1)*d'/2;
%         kernel = kernel - d*ones(1,n)/2;
%         kernel = exp(kernel);
%         kernel
    end
    
    % Create output variable
    alpha = zeros(n, 1);
    
    %Dual form (using quadratic programming via CVX)
    cvx_clear
    cvx_begin
        variable alpha(n)
        maximize(sum(alpha) -  0.5*quad_form(labels.*alpha,kernel))
        subject to
           alpha>=0
           alpha<=C
           sum(alpha.*labels)==0
    cvx_end
 
    epsilon = 0.0001;
    svii = find( alpha > epsilon & alpha < (C - epsilon));
    
    % Obtain model parameters
    %model_b = (1/length(svii))*sum(labels(svii) - kernel(svii,:)*alpha.*labels(svii));
    %model_w = data'*(alpha.*labels);

    % Obtain support vectors data
    sv = data(svii, :);
    sv_alphas = alpha(svii,:);
    sv_labels = labels(svii,:);
    sv_points = data(svii,:);
    
    % Get hyperplane vector
    model_w = sv_points'*(sv_alphas.*sv_labels);
    
    % Get hyperplane offset
    model_b = 0;
    for i = 1:size(sv_points,1)
        model_b = model_b + sv_labels(i);
        for j = 1:size(data,1)
            model_b = model_b - alpha(j) * labels(j) * kernel(svii(i), j);
        end
    end
    model_b = model_b / size(sv_points,1);
    
    % Create model structure
    model = struct('kernel', 'linear', 'w', model_w, 'b', model_b);
    model.sv = sv;
    if nargin >= 4
        model.kernel = 'rbf';
        model.sv_alphas = alpha(svii,:);
        model.sv_labels = labels(svii,:);
        model.sv_points = data(svii,:);
        model.sigma = sigma;
    end
end

