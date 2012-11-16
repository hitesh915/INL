function [ outData, transformedData, eVectors, eValues, informativeFeatures] = pca( dataMatrix, eValueThreshold)
%PCA - Principal Component Analysis implementation

    % Fill in unset optional values.
    switch nargin
        case 1
            eValueThreshold = 1;
    end
    
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
        [~, infFeatureI] = sort(abs(featuredVectors(:,i)));
        [~, infFeatureRank] = sort(infFeatureI);
        for j = 1:numFeatures
            if find(j==informativeFeatures)
                infFeatureRank(j) = 0;
            end
        end
        [~,indXMax] = max(infFeatureRank);
        informativeFeatures = [informativeFeatures, indXMax];
    end
%     [~, infFeatureI] = sort(abs(featuredVectors(:,1)));
%     [~, infFeatureRank] = sort(infFeatureI);
%     informativeFeatures = infFeatureRank;
    
end

