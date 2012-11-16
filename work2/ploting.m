function [ ] = ploting( K, meansAccuracyList, SEMList )
%PLOTING Summary of this function goes here
%   Detailed explanation goes here

    figure;
    hold on;
    %errorbar([K,K,K], meansAccuracyList(1,:), SEMList(1, :).*2.262);
    
    l1 = line(K, meansAccuracyList(1,1:7));
    e1 = errorbar(K, meansAccuracyList(1,1:7), SEMList(1,1:7).*2.262);
    
    set(l1,'Color', 'r', 'Marker', '.');
    set(e1,'Color', 'r', 'LineStyle', 'none');
    
    
    hold on;
    
    l2 = line(K, meansAccuracyList(1,8:14));
    e2 = errorbar(K, meansAccuracyList(1,8:14), SEMList(1,8:14).*2.262);
    set(l2,'Color', 'g', 'Marker', 'x');
    set(e2,'Color', 'g', 'LineStyle', 'none');
    hold on;
    
    l3 = line(K, meansAccuracyList(1,15:21));
    e3 = errorbar(K, meansAccuracyList(1,15:21), SEMList(1,15:21).*2.262);
    set(l3,'Color', [0,0,0.5], 'Marker', 's');
    set(e3,'Color', [0,0,0.5], 'LineStyle', 'none');
    legend('r=1', '', 'r=2', '', 'r=3');

    grid on;

    xlabel('K value');
    ylabel('Accuracy');

end

