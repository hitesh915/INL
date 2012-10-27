function [ outData, transformedData, eVectors, eValues] = pca( dataMatrix, eValueThreshold)
%PCA - Principal Component Analysis implementation
    
    %Calculate the covariance matrix
    covMatrix = cov(dataMatrix);
    
    %Calculate the eigenvectors and eigenvalues of the covariance matrix
    [eVectors, eValues] = eig(covMatrix);
    
    %Choose components
    %order the eValues and eVectors in decend order
    eValues = flipud(diag(eValues));
    
    eVectors = fliplr(eVectors);
    
    %construt a new feature vector based on the eValueThreshold
    featureVector = eVectors(:, eValues >= eValueThreshold);
    
    %Derive the new data set
    transformedData = featureVector.' * dataMatrix.';
    
    %Reconstruct the old data back
    rowData = featureVector * transformedData;
    outData = rowData.';
    
end

