function [ output_args ] = plotAdaboost(model, data, labels)
    points1 = data(labels==1,:);
    h1 = scatter(points1(:,1), points1(:,2), 'r');
    hold on;
    
    points2 =  data(labels==-1,:);
    h2 = scatter(points2(:,1), points2(:,2), 'g');
    hold on;
    
    % Plot perceptrons
    for i = 1:model.T
        weights = model.models(i,:)';
        offset  = model.offsets(i);
        
        % Get points
        x = [-1 1];
        yPart = (offset - x * weights(1))/weights(2);
        
        % Draw lines
        plot(x, yPart);
    end
end

