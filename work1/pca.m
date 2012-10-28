function [ outData, transformedData, eVectors, eValues, informativeFeatures] = pca( dataMatrix, eValueThreshold)
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
    featuredVectors = eVectors(:, eValues >= eValueThreshold);
    
    %Derive the new data set
    transformedData = featuredVectors.' * dataMatrix.';
    
    %Reconstruct the old data back
    rowData = featuredVectors * transformedData;
    outData = rowData.';
    
    %Informative features
    [numFeatures, numFeaturedVectors] = size(featuredVectors);
    featureRows = 1:numFeatures;
    informativeFeatures = [];
    for i = 1:numFeaturedVectors
        [~,infFeatureI] = max(abs(featuredVectors(setdiff(featureRows,informativeFeatures),i)));
        informativeFeatures = [informativeFeatures, (infFeatureI+sum(informativeFeatures < infFeatureI))];
    end
    
end

