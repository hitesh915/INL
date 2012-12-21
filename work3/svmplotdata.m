function [] = svmplotdata(x,label,svmStruct)
%Plot the training data representing the its class the supporvectors used
%and the classification boundary.

    points1 = x(label==1,:);
    h1 = plot(points1(:,1),points1(:,2),'Marker','+','LineStyle','none','Color',[1 0 0],...
    'DisplayName','Class -1');
    hAxis = get(h1,'parent');
    hold on
    pointsM1 =  x(label==-1,:);
    h2 = plot(pointsM1(:,1),pointsM1(:,2),'MarkerFaceColor',[0.0392156876623631 0.141176477074623 0.415686279535294],...
    'MarkerSize',3,'Marker','o','LineStyle','none','Color',[0.0392156876623631 0.141176477074623 0.415686279535294],'DisplayName','Class 1');

    %rescale support vectors
    sv = svmStruct.sv_points;
    sv(:,1) = sv(:,1)*svmStruct.stdTrain(1)+svmStruct.meanTrain(1);
    sv(:,2) = sv(:,2)*svmStruct.stdTrain(2)+svmStruct.meanTrain(2);
    h3 = plot(sv(:,1), sv(:,2),'MarkerSize',8,'Marker','o','LineStyle','none','DisplayName','Support Vector','Color',[0 0 0]);

    lims = axis(hAxis);
    [X,Y] = meshgrid(linspace(lims(1),lims(2)),linspace(lims(3),lims(4)));
    Xorig = X; Yorig = Y;

    [~, Z] = test_svm([X(:),Y(:)],svmStruct);

    contour(Xorig,Yorig,reshape(Z,size(X)),[0 0],'LineStyle','--','LineColor',[0 0 0],'LevelList',0,'DisplayName','Class boundary');

    drawnow

hLines = [h1,h2];