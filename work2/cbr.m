function [ accuracy ] = cbr( trainMatrix, testMatrix, K, r, knn_type )
    %CBR This function implements the case-based reasoining algorithm for
    %  the kNN, selected kNN and featured kNN algorithms, classifying the
    %  instances of testMatrix by using the trainMatrix values as neighbors
    %
    %  INPUTS:
    %    trainMatrix = Matrix with the training data used by the classifier
    %    testMatrix = Matrix with the test data to classify
    %    K = K value to use as number of neighbors in the kNN algorithm
    %    r = Operator to use to calculate the distance between individuals
    %    knn_type = Type of kNN algorithm to apply:
    %      - 1 = kNN
    %      - 2 = weighted kNN
    %      - 3 = selected kNN
    %
    %  OUTPUTS:
    %    accuracy = Proportion of hits (between 0 and 1) of the
    %    classification process for the testMatrix matrix

    function [ entropy ] = get_entropy( countsVector )
        vector = countsVector(countsVector ~= 0);
        
        nElements = sum(vector);
        nClasses = size(vector);
        nClasses = nClasses(2);
        
        if nClasses == 1
            entropy = 0;
            return;
        end
        
        entropy = 0;
        for c=1:nClasses
            p = vector(c)/nElements;
            entropy = entropy - p * log2(p);
        end
        entropy = entropy / log2(nClasses);
    end
        
    % Get number of instances on the test matrix
    tmSize = size(testMatrix);
    tmSize = tmSize(1);
    
    % Initialize classification success counter
    numSuccess = 0;
    
    %standarize data
    [stdAttributes, mean, stdDev] = standarizer(trainMatrix(:,1:end-1));
    stdTrain = [stdAttributes, trainMatrix(:,end)];
    
    if knn_type > 1
        %Get the weights using FS filter relieff
        [~, weights] = relieff(stdTrain(:,1:end-1),stdTrain(:,end),K, 'method', 'classification', 'categoricalx', 'off');

        %ranging the weights into the interval [0,1]
        weights = (weights - (-1))./2;

        % If selected kNN, remove unused attributes
        if knn_type == 3
            threshold = sum(weights)/size(weights,2);
            selector = weights >= threshold;
            stdTrain = [ stdTrain(:, selector), stdTrain(:, end) ];
            mean = mean(:, selector);
            stdDev = stdDev(:, selector);
        end
    end
    
    % Classify the test individuals
    for i=1:tmSize
        % Get instance to classify
        instance = testMatrix(i,:);
        if knn_type == 3
            stdInstance = (instance(:,selector)-mean)./stdDev; 
        else
            stdInstance = (instance(:,1:end-1)-mean)./stdDev;
        end
        stdInstance(isnan(stdInstance)) = 0;
        stdInstance = [stdInstance, instance(:,end)];
        
        % Get the predictors
        if knn_type == 1
            [predictors, distances] = kNN(stdTrain, stdInstance(:,1:end-1), K, r);
        elseif knn_type == 2
            [predictors, distances] = weightedKNN(stdTrain, stdInstance(:,1:end-1), K, r, weights);
        elseif knn_type == 3
            [predictors, distances] = kNN(stdTrain, stdInstance(:,1:end-1), K, r);
        end    
        
        % Generate counter vector for the classes
        countClasses = zeros(1,max(predictors(:,end)));

        % Count instances
        for j=1:K
            cls = predictors(j, end);
            countClasses(cls) = countClasses(cls) + 1;
        end
        
        % Get maximum count
        mcount = max(countClasses);
        
        % If conflict, get class with the predictors closer to the instance
        numConflicting = sum(countClasses == mcount);
        if numConflicting > 1
            % Get conflicting classes indexs
            clsConflicting = find(countClasses == mcount);
            
            % Get class with the shortest sum of distances
            best_dist = NaN;
            best_case = NaN;
            for j=1:numConflicting
                % Calculate total distance from predictors to instance
                tfilter = predictors(:,end) == clsConflicting(j);
                tdistances  = distances(tfilter);
                dists = sum(tdistances);
                
                if isnan(best_dist) || dists < best_dist
                    best_dist = dists;
                    best_case = clsConflicting(j);
                end
            end
            
            % Select best found as the class
            class = best_case;
            
        % If no conflicts, get most polled class
        else
            class = find(countClasses == mcount);
        end
        
        % If class is the expected one
        if class == instance(end)
            % Increase success counter
            numSuccess = numSuccess + 1;
            
            % If there was conflict, learn example
            if get_entropy(countClasses) >= 0.9
                trainMatrix = [trainMatrix ; instance];
                
                %RE-standarize data
                [stdAttributes, mean, stdDev] = standarizer(trainMatrix(:,1:end-1));
                stdTrain = [stdAttributes, trainMatrix(:,end)];
                
                if knn_type > 1
                    %Get the weights using FS filter relieff
                    [~, weights] = relieff(stdTrain(:,1:end-1),stdTrain(:,end),K, 'method', 'classification', 'categoricalx', 'off');

                    %ranging the weights into the interval [0,1]
                    weights = (weights - (-1))./2;

                    % If selected kNN, remove unused attributes
                    if knn_type == 3
                        threshold = sum(weights)/size(weights,2);
                        selector = weights >= threshold;
                        stdTrain = [ stdTrain(:, selector), stdTrain(:, end) ];
                        mean = mean(:, selector);
                        stdDev = stdDev(:, selector);
                    end
                end
            end
        end
    end
    
    accuracy = numSuccess / tmSize;
end