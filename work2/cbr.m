function [ accuracy ] = cbr( trainMatrix, testMatrix, K, r )
    %CBR Summary of this function goes here
    %   Detailed explanation goes here

%     function [ entropy ] = get_entropy( countsVector )
%         vector = countsVector(countsVector ~= 0);
%         
%         nElements = sum(vector);
%         nClasses = size(vector);
%         nClasses = nClasses(2);
%         
%         if nClasses == 1
%             entropy = 0;
%             return;
%         end
%         
%         entropy = 0;
%         for i=1:nClasses
%             p = vector(i)/nElements;
%             entropy = entropy - p * log2(p);
%         end
%         entropy = entropy / log2(nClasses);
%     end
        
    % Get number of instances on the test matrix
    tmSize = size(testMatrix);
    tmSize = tmSize(1);
    
    % Initialize classification success counter
    numSuccess = 0;
    
    % Classify the test individuals
    for i=1:tmSize
        % Get instance and predictors
        instance = testMatrix(i,:);
        predictors = kNN(trainMatrix, instance(:,1:end-1), K, r);
        
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
                % Get predictors of the class
                tpredictors = predictors(predictors(:,end) == clsConflicting(j), :);
            
                % Calculate total distance from predictors to instance
                dists = pdist2(tpredictors(:,1:end-1), instance(:,1:end-1), 'minkowski',r);
                dists = sum(dists);
                
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
            if numConflicting > 1
                trainMatrix = [trainMatrix ; instance];
            end
        end
    end
    
    accuracy = numSuccess / tmSize;
end