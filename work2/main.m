%The dataset used on the experiment
dataset = 'iris';

%Several values of K
K = [1,3,5,7,9,11,13];

%Several values of R
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
        
        %Execute the cbr for each fold
        for fold = 1:10
            train = data{fold,1};
            test = data{fold,2};
            accuracies(fold) = cbr( train, test, k, r , 1);
        end
        
        %Compute the statistics
        accuracyMean = mean(accuracies);
        accuracySTD = std(accuracies);
        accuracySEM = accuracySTD/sqrt(size(K,2));
        
        %Print informaption about the results of the execution
        fprintf(strcat('K:\t',num2str(k),'\nR:\t', num2str(r),'\n'));
        fprintf(strcat('Mean accuracy:\t\t', num2str(accuracyMean), '\n'));
        fprintf(strcat('Accuracy standard dev.:\t', num2str(accuracySTD), '\n'));
        fprintf(strcat('Standard Error of Mean:\t', num2str(accuracySEM), '\n'));
        
        %Add the info to the statistics list
        meansAccuracyList = [meansAccuracyList, accuracyMean];
        stdAccuracyList = [stdAccuracyList, accuracySTD];
        SEMList = [SEMList, accuracySEM];
    end
end

%Plot the results
ploting(dataset, K, meansAccuracyList, SEMList);
 
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
    kAccuracies(fold) = cbr(train, test, k, r, 1);
    wAccuracies(fold) = cbr(train, test, k, r, 2);
    sAccuracies(fold) = cbr(train, test, k, r, 3);
end

% Calculate mean accuracy for the kNN algorithm
kAccuracyMean = mean(kAccuracies);
kAccuracySTD = std(kAccuracies);
kAccuracySEM = kAccuracySTD/sqrt(size(K,2));

% Show results for kNN
fprintf('kNN:');
fprintf(strcat('Mean accuracy:\t\t', num2str(kAccuracyMean), '\n'));
fprintf(strcat('Accuracy standard dev.:\t', num2str(kAccuracySTD), '\n'));
fprintf(strcat('Standard Error of Mean:\t', num2str(kAccuracySEM), '\n'));

% Calculate mean accuracy for the weighted kNN algorithm
wAccuracyMean = mean(wAccuracies);
wAccuracySTD = std(wAccuracies);
wAccuracySEM = wAccuracySTD/sqrt(size(K,2));

% Show results for weighted kNN
fprintf('Weighted kNN:');
fprintf(strcat('Mean accuracy:\t\t', num2str(wAccuracyMean), '\n'));
fprintf(strcat('Accuracy standard dev.:\t', num2str(wAccuracySTD), '\n'));
fprintf(strcat('Standard Error of Mean:\t', num2str(wAccuracySEM), '\n'));

% Calculate mean accuracy for the selected kNN algorithm
sAccuracyMean = mean(sAccuracies);
sAccuracySTD = std(sAccuracies);
sAccuracySEM = sAccuracySTD/sqrt(size(K,2));

% Show results for segmented kNN
fprintf('Selected kNN:');
fprintf(strcat('Mean accuracy:\t\t', num2str(sAccuracyMean), '\n'));
fprintf(strcat('Accuracy standard dev.:\t', num2str(sAccuracySTD), '\n'));
fprintf(strcat('Standard Error of Mean:\t', num2str(sAccuracySEM), '\n'));
