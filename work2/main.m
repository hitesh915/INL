dataset = 'pen-based';

K = [1,3,5,7,9,11,13];
R = 1:3;

meansAccuracyList = [];
stdAccuracyList = [];
SEMList = [];

% Get the data from files
data = [];
for fold = 1:10
   [train, test] = parser_nfold(dataset, fold);
   data = [data;{train,test}];
end

% STEP 1: TEST DIFFERENT K AND R VALUES
% --------------------------------------------------------------

% Test different K and R values
for r = R
    for k = K
        accuracies = zeros(1, 10);
        for fold = 1:10
            train = data{fold,1};
            test = data{fold,2};
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


% STEP 2: APPLY WEIGHTED AND SELECTED KNN ALGORITHMS
% --------------------------------------------------------------

% Set K and R values to use
k = 1;
r = 1;

% Initialize accuracy vectors
wAccuracies = zeros(1, 10);
sAccuracies = zeros(1, 10);

% Classify folds with the algorithms
for fold = 1:10
    train = data{fold,1};
    test = data{fold,2};
    wAccuracies(fold) = cbr(train, test, k, r, 2);
    sAccuracies(fold) = cbr(train, test, k, r, 3);
end

% Calculate mean accuracy for the weighted kNN algorithm
wAccuracyMean = mean(wAccuracies);
wAccuracySTD = std(wAccuracies);
wAcuracySEM = wAccuracySTD/sqrt(size(K,2));

% Show results for weighted kNN
fprintf('Weighted kNN:');
fprintf(strcat('Mean accuracy:\t\t', num2str(wAccuracyMean), '\n'));
fprintf(strcat('Accuracy standard dev.:\t', num2str(wAccuracySTD), '\n'));
fprintf(strcat('Standard Error of Mean:\t', num2str(wAccuracySEM), '\n'));

% Calculate mean accuracy for the selected kNN algorithm
sAccuracyMean = mean(sAccuracies);
sAccuracySTD = std(sAccuracies);
sAcuracySEM = sAccuracySTD/sqrt(size(K,2));

% Show results for segmented kNN
fprintf('Selected kNN:');
fprintf(strcat('Mean accuracy:\t\t', num2str(sAccuracyMean), '\n'));
fprintf(strcat('Accuracy standard dev.:\t', num2str(sAccuracySTD), '\n'));
fprintf(strcat('Standard Error of Mean:\t', num2str(sAccuracySEM), '\n'));