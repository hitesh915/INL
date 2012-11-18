function [ kPredictors, kDistances ] = selectedKNN(TrainMatrix, current_instance, K, r)
    % Get matrix & instance with selected features
    [~, ~, ~, ~, featured] = pca(TrainMatrix(:,end-1), 0.1);
    features = TrainMatrix(:,featured);
    instance = current_instance(:,featured);
    
    %Calculate the minkowski distance from current_instance to predictors
    distances = pdist2(features, instance, 'minkowski', r);
 
    [~, distIndex] = sort(distances);
    [~, distRank] = sort(distIndex);
    
    iPredictors = distRank <= K;
    kPredictors = TrainMatrix(distRank <= K,:);
    kDistances = distances(iPredictors);
end

