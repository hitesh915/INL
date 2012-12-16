function [hAxis,hLines] = svmplotdata(x,label,svmStruct)
class1 = 'r+';
class2 = 'g*';
svStyle = 'bo';

points1 = x(label==1,:);
h1 = plot(points1(:,1),points1(:,2),class1);
hAxis = get(h1,'parent');
hold on
pointsM1 =  x(label==-1,:);
h2 = plot(pointsM1(:,1),pointsM1(:,2),class2);

%rescale support vectors
sv = svmStruct.sv;
sv(:,1) = sv(:,1)*svmStruct.stdTrain(1)+svmStruct.meanTrain(1);
sv(:,2) = sv(:,2)*svmStruct.stdTrain(2)+svmStruct.meanTrain(2);
h3 = plot(sv(:,1), sv(:,2),svStyle);

%axis equal
drawnow
% reset hold state if it was off

hLines = [h1,h2];