function [model] = train_svm(labels, data, C, sigma)
%TRAIN_SVM train a SVM with the given data
%   INPUT:
%       - labels: the labels of each instance of the training set.
%       - data: the dataset used to train the SVM
%       - C: The soft marging parameter
%       - sigma: the sigma value used in the gausian RBF funtion. If this
%       value is not fixed the algorithm will apply a linear Kernel.
%   OUTPUT:
%       - model: an structure of the model that contais all the parameters
%       needed to perform the classification. See below.

    %Calc the mean and the standard deviation of the data
    mean = nanmean(data);
    std = nanstd(data);

    %Standarize the data
    data = standarizer(data);
    
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
        kernel = rbfKernel(data, data, sigma);
    end
    
    % Create output variable
    alpha = zeros(n, 1);
    
    %Dual form (using quadratic programming via CVX)
    cvx_clear
    cvx_quiet(true)
    cvx_begin
        variable alpha(n)
        maximize(alpha'*ones(n,1)-0.5*alpha'*(diag(labels)*kernel*diag(labels))*alpha)
        subject to
           0<= alpha <=C
           sum(alpha.*labels)==0
    cvx_end
    
    %Get the index of the support vectors
    svii = find( alpha > sqrt(eps));

    % Obtain support vectors data
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
    model = struct('kernel', 'linear', 'w', model_w, 'b', model_b, 'c', C);
    model.sv_points = sv_points;
    model.meanTrain = mean;
    model.sv_alphas = sv_alphas;
    model.sv_labels = sv_labels;
    model.stdTrain = std;
    model.svii = svii;
    if nargin >= 4
        model.kernel = 'rbf';
        model.sigma = sigma;
    end
end

