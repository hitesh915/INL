function [ output] = mds( matrix, dimensions)
% Implements the multidimensional scaling algorithm (MDS).
%   inputs: a matrix with all the elements, and the desired output
%   dimensions
%   output: a representation in d dimensions of the data

%Compute the matrix distance
distanceMatrix = pdist2(matrix,matrix);

%Compute the SQUARED matrix distance
squaredDistances = distanceMatrix.*distanceMatrix;

%Apply the double centering
J = eye(length(matrix)) - (length(matrix)^-1)*ones(length(matrix));
B = (-1./2)*J*squaredDistances*J;

%Compute the eigenvalues an eigenvectors
[eVectors, eValues] = eig(B);

%Compute the final output.
output = eVectors(:,1:dimensions)*(eValues(1:dimensions,1:dimensions).^(1/2));

end

