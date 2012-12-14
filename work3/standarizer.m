function [ matrixZ, meanC, stdDevC ] = standarizer( matrix )
    %STANDARIZERS Returns a standarized version of the data with the missing
    %  values substituted with the mean of the attribute
    %
    %  INPUTS:
    %    matrix = Matrix with the data to standarize
    %
    %  OUTPUTS:
    %    matrixZ = Original matrix standarized
    %    meanC = Vector with the mean of each column
    %    stdDevC = Vector with the standard deviation of each column

    %Compute the mean & standard deviation of each column (NaNs no compute)
    meanC = nanmean(matrix);
    stdDevC = nanstd(matrix);
    
    %Logical matrix with ones in the place of NaNs and 0 the rest
    matrixNaNs = isnan(matrix);
    
    %Inter. matrix, with the mean in the place of NaNs
    matrixIntermediate = matrixNaNs * diag(meanC);
    
    %Join inter. matrix with original matrix. Substitude missing
    %values(NaNs) for the mean of the attribute
    matrix(matrixNaNs) = matrixIntermediate(matrixNaNs);
   
    %Get a standarized matrix
    matrixZ = zscore(matrix);

end

