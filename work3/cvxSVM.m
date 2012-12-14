kernel = data*data';
C = 2;
n = 40;

%Dual form
cvx_clear
cvx_begin
    variables alpha(n)
    maximize( sum(alpha) -  0.5*quad_form(labels.*alpha,kernel))
    subject to
       alpha>0
       alpha<C
       sum(alpha.*labels)==0
cvx_end