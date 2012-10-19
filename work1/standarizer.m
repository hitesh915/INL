function [ std_matrix ] = standarizer( matrix )
    [nRows nCols] = size(matrix);
    std_matrix = zeros(nRows, nCols);
    
    for i = 1:nCols
        column = matrix(:,i);
        lsNan = isnan(column);

        % Remove NaNs
        indices = find(lsNan);
        column(find(lsNan)) = [];
        
        % Number of elements and NaNs
        nmNan = sum(lsNan);
        nmNum = numel(column);
        
        % Calculate mean and standard deviation
        mean = sum(column) / nmNum;
        stdv = sum((column-mean).^2) / (nmNum - 1);
        
        % Standarize column
        for j = 1:nRows
            if lsNan(j)
                std_matrix(j,i) = 0;
            else
                std_matrix(j,i) = (matrix(j,i) - mean) / stdv;
            end
        end
    end
end

