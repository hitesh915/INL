function [model] = train_svm(labels, data, C, sigma)
%TRAIN_SVM Summary of this function goes here
%   Detailed explanation goes here
    
    % If C is undefined, set C to 2
    if nargin < 3
       C = 2;
    end
    
    % Count number of data points
    n = size(data, 1);
    
    % Define linear/RBF kernel
    if nargin < 4
        kernel = data*data';
    else
        kernel = data*data' / sigma^2;
        d = diag(kernel);
        kernel = kernel - ones(n,1)*d'/2;
        kernel = kernel - d*ones(1,n)/2;
        kernel = exp(kernel);
    end
    
    % Create output variable
    alpha = zeros(n, 1);
    
    %Dual form (using quadratic programming via CVX)
    cvx_clear
    cvx_begin
        variables alpha(n)
        maximize(sum(alpha) -  0.5*quad_form(labels.*alpha,kernel))
        subject to
           alpha>=0
           alpha<=C
           sum(alpha.*labels)==0
    cvx_end
 
    epsilon = 0.0001;
    svii = find( alpha > epsilon & alpha < (C - epsilon));
    
    % Obtain model parameters
    model_b = (1/length(svii))*sum(labels(svii) - kernel(svii,:)*alpha.*labels(svii));
    model_w = data'*(alpha.*labels);

    % Obtain support vectors
    sv = data(alpha > epsilon & alpha < (C - epsilon), :);
    
    % Create model structure
    model = struct('kernel', 'linear', 'w', model_w, 'b', model_b);
    model.sv = sv;
    if nargin >= 4
        model.kernel = 'rbf';
        model.sigma = sigma;
        model.data = data;
    end
end

