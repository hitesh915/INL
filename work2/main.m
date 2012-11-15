dataset = 'bal';

K = 1:10;
R = 1:3;

meansAccuracyList = [];
stdAccuracyList = [];


for r = R
    for k = K
        accuracies = zeros(1, 10);
        for fold = 1:10
            [train, test] = parser_nfold(dataset, fold);
            accuracies(fold) = cbr( train, test, k, r , 1);
        end
        fprintf(strcat('K:\t',num2str(k),'\nR:\t', num2str(r),'\n'));
        fprintf(strcat('Mean accuracy:\t\t', num2str(mean(accuracies)), '\n'));
        fprintf(strcat('Accuracy standard dev.:\t', num2str(std(accuracies)), '\n'));
        meansAccuracyList = [meansAccuracyList, mean(accuracies)];
        stdAccuracyList = [stdAccuracyList, std(accuracies)];
    end
end

figure;
plot(K, meansAccuracyList(1,1:10),'-', K, meansAccuracyList(1,11:20),'r-', K, meansAccuracyList(1,21:30),'g-');
legend('r=1', 'r=2', 'r=3');

grid on;

xlabel('K value');
ylabel('Accuracy');