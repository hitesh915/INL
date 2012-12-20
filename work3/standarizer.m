function [ matrixZ, meanC, stdDevC ] = standarizer( matrix, mean, stdDev )
    %STANDARIZERS Returns a standarized version of the data with the missing
    %  values substituted with the mean of the attribute
    %
    %  INPUTS:
    %    matrix = Matrix with the data to standarize
    %    mean(optional) = the mean used to standarize
    %    stdDev(optional) = the standard deviation used to standarize
    %
    %  OUTPUTS:
    %    matrixZ = Original matrix standarized
    %    meanC = Vector with the mean of each column
    %    stdDevC = Vector with the standard deviation of each column

    %Compute the mean & standard deviation of each column (NaNs no compute)
    if nargin < 3
        meanC = nanmean(matrix);
        stdDevC = nanstd(matrix);
    else
        meanC = mean;
        stdDevC = stdDev;
    end
    
    %Logical matrix with ones in the place of NaNs and 0 the rest
    
    matrixZ = bsxfun(@minus, matrix, meanC);
    matrixZ = bsxfun(@rdivide, matrixZ, stdDevC);
    matrixZ(isnan(matrixZ)) = 0;

end

