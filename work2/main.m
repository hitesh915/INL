dataset = 'waveform';

K = [1,3,5,7,9,11,13];
R = 1:3;

meansAccuracyList = [];
stdAccuracyList = [];
SEMList = [];


for r = R
    for k = K
        accuracies = zeros(1, 10);
        for fold = 1:10
            [train, test] = parser_nfold(dataset, fold);
            accuracies(fold) = cbr( train, test, k, r , 1);
        end
        accuracyMean = mean(accuracies);
        accuracySTD = std(accuracies);
        accuracySEM = accuracySTD/sqrt(size(K,2));
        fprintf(strcat('K:\t',num2str(k),'\nR:\t', num2str(r),'\n'));
        fprintf(strcat('Mean accuracy:\t\t', num2str(accuracyMean), '\n'));
        fprintf(strcat('Accuracy standard dev.:\t', num2str(accuracySTD), '\n'));
        fprintf(strcat('Standard Error of Mean:\t', num2str(accuracySEM), '\n'));
        meansAccuracyList = [meansAccuracyList, accuracyMean];
        stdAccuracyList = [stdAccuracyList, accuracySTD];
        SEMList = [SEMList, accuracySEM];
    end
end

ploting(K, meansAccuracyList, SEMList);