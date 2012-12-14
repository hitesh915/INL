%data = data';
rbf_kernel = false;

% If RBF kernel (use sig parameter)
if rbf_kernel
    n = size(data, 1);
    kernel = data*data' / sig^2;
    d = diag(K);
    kernel = kernel - ones(n,1)*d'/2;
    kernel = kernel - d*ones(1,n)/2;
    kernel = exp(K);
    
% If linear kernel
else
    kernel = data*data';
end

C = 2;
n = size(data, 1);

%Dual form
cvx_clear
cvx_begin
    variables alpha(n)
    maximize( sum(alpha) -  0.5*quad_form(labels.*alpha,kernel))
    subject to
       alpha>=0
       alpha<=C
       sum(alpha.*labels)==0
cvx_end

