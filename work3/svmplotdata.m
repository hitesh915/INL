function [] = svmplotdata(x,label,svmStruct)
%Plot the training data representing the its class the supporvectors used
%and the classification boundary.

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
    sv = svmStruct.sv_points;
    sv(:,1) = sv(:,1)*svmStruct.stdTrain(1)+svmStruct.meanTrain(1);
    sv(:,2) = sv(:,2)*svmStruct.stdTrain(2)+svmStruct.meanTrain(2);
    h3 = plot(sv(:,1), sv(:,2),svStyle);

    lims = axis(hAxis);
    [X,Y] = meshgrid(linspace(lims(1),lims(2)),linspace(lims(3),lims(4)));
    Xorig = X; Yorig = Y;

    [~, Z] = test_svm([X(:),Y(:)],svmStruct);

    contour(Xorig,Yorig,reshape(Z,size(X)),[0 0],'k');

    drawnow

hLines = [h1,h2];