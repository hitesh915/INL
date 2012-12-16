function hSV = svmplotsvs(hAxis,hLines,groupString,svm_struct)
%SVMPLOTSVS Plot the support vectors and the separating line for SVMTRAIN

%   Copyright 2004-2010 The MathWorks, Inc.
%   $Revision: 1.1.12.6.14.2 $  $Date: 2011/03/17 22:26:00 $

hold on;
sv = svm_struct.sv;
% see if we need to unscale the data

for c = 1:size(sv, 2)
        sv(:,c) = (sv(:,c)*svm_struct.stdTrain(c)) + svm_struct.meanTrain(c);
end

% plot support vectors
hSV = plot(sv(:,1),sv(:,2),'ko');

lims = axis(hAxis);
[X,Y] = meshgrid(linspace(lims(1),lims(2)),linspace(lims(3),lims(4)));
Xorig = X; Yorig = Y;

% need to scale the mesh
X = svm_struct.stdTrain(1) * (X + svm_struct.meanTrain(1));
Y = svm_struct.stdTrain(2) * (Y + svm_struct.meanTrain(2));

[~, Z] = test_svm(svm_struct,[X(:),Y(:)]); 

Z

contour(Xorig,Yorig,reshape(Z,size(X)),[0 0],'k');
hold off;
labelString = cellstr(groupString);
labelString{end+1} = 'Support Vectors';
legend([hLines(1),hLines(2),hSV], labelString);