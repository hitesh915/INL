function [model] = train_adaboost(labels, data, T)
%TRAIN_SVM Summary of this function goes here
%   Detailed explanation goes here

    % Standarize data
    mean = nanmean(data);
    std = nanstd(data);
    data = standarizer(data);

    function [clabels, error, h] = select_weighted_threshold(data, labels, weights)
        function y = classify_data(h, x)
            if(h.direction == 1)
                y =  double(x(:,h.dimension) >= h.threshold);
            else
                y =  double(x(:,h.dimension) < h.threshold);
            end
            y(y==0) = -1;
        end
        
        % Number of treshold steps
        ntre=2e5;

        % Split data & weights in two classes
        r1 = data(labels<0,:); w1 = weights(labels<0);
        r2 = data(labels>0,:); w2 = weights(labels>0);

        % Calculate the min and max for every dimensions
        minr=min(data,[],1)-1e-10; maxr=max(data,[],1)+1e-10;

        % Make a weighted histogram of the two classes
        p2c= ceil((bsxfun(@rdivide,bsxfun(@minus,r2,minr),(maxr-minr)))*(ntre-1)+1+1e-9);   p2c(p2c>ntre)=ntre;
        p1f=floor((bsxfun(@rdivide,bsxfun(@minus,r1,minr),(maxr-minr)))*(ntre-1)+1-1e-9);  p1f(p1f<1)=1;
        ndims=size(data,2);
        i1=repmat(1:ndims,size(p1f,1),1);  i2=repmat(1:ndims,size(p2c,1),1);
        h1f=accumarray([p1f(:) i1(:)],repmat(w1(:),ndims,1),[ntre ndims],[],0);
        h2c=accumarray([p2c(:) i2(:)],repmat(w2(:),ndims,1),[ntre ndims],[],0);

        % Calculate error for every all possible treshold and dimension
        h2ic=cumsum(h2c,1);
        h1rf=cumsum(h1f(end:-1:1,:),1); h1rf=h1rf(end:-1:1,:);
        e1a=h1rf+h2ic;
        e2a=sum(weights)-e1a;

        % Select treshold value and dimension with the minimum error
        [err1a,ind1a] = min(e1a,[],1);  dim1a = (1:ndims); dir1a = ones(1,ndims);
        [err2a,ind2a] = min(e2a,[],1);  dim2a = (1:ndims); dir2a = -ones(1,ndims);
        A = [err1a(:),dim1a(:),dir1a(:),ind1a(:);err2a(:),dim2a(:),dir2a(:),ind2a(:)];
        
        [error,k] = min(A(:,1)); 
        dim=A(k,2); 
        dir=A(k,3); 
        ind=A(k,4);
        
        thresholds = linspace(minr(dim),maxr(dim),ntre);
        thr = thresholds(ind);

        % Save new threshold
        h = struct;
        h.dimension = dim; 
        h.threshold = thr; 
        h.direction = dir;
        
        % Calculate labels
        clabels = classify_data(h, data);
    end

    % Get size of training data
    m = size(data, 1);
    
    % Initialize perceptrons matrix
    models = struct;
    
    % Initialize vector of weights
    D = zeros(m,1);
    D(:) = 1/m;
    
    for ii=1:T
        % Train new classifier
        [results, error, h] = select_weighted_threshold(data, labels, D);
        
        % If error >= 1/2, stop algorithm
        if error >= 0.5
            break
        end
        
        % Calculate alpha
        alpha = 1/2 * log((1 - error) / (max(error, eps)));
        
        % Store the model parameters
        models(ii).alpha = alpha;
        models(ii).dimension = h.dimension;
        models(ii).threshold = h.threshold;
        models(ii).direction = h.direction;

        % Update vector of weights
        D = D .* exp(-models(ii).alpha .* labels .* results);
        D = D ./ sum(D);
    end
    
    % Prepare response model
    model = struct;
    model.models = models;
    model.mean = mean;
    model.std = std;
end

