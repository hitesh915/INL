%The dataset used on the experiment
dataset = 'pen-based';

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
        fprintf('K:\t%d\nR:\t%d\n',r,k);
        fprintf('Mean accuracy:\t%.4f\n',accuracyMean);
        fprintf('Accuracy standard dev.:\t%.4f\n',accuracySTD);
        fprintf('Standard Error of Mean:\t%.4f\n\n',accuracySEM);
        
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
[~, idxMax] = max(meansAccuracyList);
idxK = mod(idxMax,7);
if idxK == 0
    idxK = 7;
end
k = K(idxK);
r = ceil(idxMax/7);

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
fprintf('COMPARATIVE (K = %d R= %d)\n----------------------\n\nkNN:\n',k,r);
fprintf('Mean accuracy:\t\t%.4f\n',kAccuracyMean);
fprintf('Accuracy standard dev.:\t%.4f\n',kAccuracySTD);
fprintf('Standard Error of Mean:\t%.4f\n',kAccuracySEM);

% Calculate mean accuracy for the weighted kNN algorithm
wAccuracyMean = mean(wAccuracies);
wAccuracySTD = std(wAccuracies);
wAccuracySEM = wAccuracySTD/sqrt(size(K,2));

% Show results for weighted kNN
fprintf('\nWeighted kNN:\n');
fprintf('Mean accuracy:\t\t%.4f\n',wAccuracyMean);
fprintf('Accuracy standard dev.:\t%.4f\n',wAccuracySTD);
fprintf('Standard Error of Mean:\t%.4f\n',wAccuracySEM);

% Calculate mean accuracy for the selected kNN algorithm
sAccuracyMean = mean(sAccuracies);
sAccuracySTD = std(sAccuracies);
sAccuracySEM = sAccuracySTD/sqrt(size(K,2));

% Show results for segmented kNN
fprintf('\nSelected kNN:\n');
fprintf('Mean accuracy:\t\t%.4f\n',sAccuracyMean);
fprintf('Accuracy standard dev.:\t%.4f\n',sAccuracySTD);
fprintf('Standard Error of Mean:\t%.4f\n',sAccuracySEM);


fprintf('\nSTUDENT PAIRED T-TEST\n=========================\n');

%Perform a paired t-test comparing simpleKnn with weightedKnn, with a
%significance of 0.05.
fprintf('\nPaired T test (alpha=0.05): CBR(kNN) vs CBR(weightedkNN)\n---------------------------------------------------------\nH0:\tkNN == weightedKNN\nH1:\tkNN != weightedKNN\n');

[simpleVsWeightedH,simpleVsWeightedPValue, simpleVsWeightedCI]  = ttest(kAccuracies, wAccuracies, 0.05, 'both', 2);

fprintf('\nP-value:\t%.4f\n',simpleVsWeightedPValue);
fprintf('Confidence interval:\t[%.4f, %.4f]\n',simpleVsWeightedCI(1),simpleVsWeightedCI(2));

if simpleVsWeightedH == 1
    if simpleVsWeightedCI(2) < 0
        fprintf('Result: H0 rejection! weightedKNN significantly better than kNN\n');
    else
        fprintf('Result: H0 rejection! kNN significantly better than weightedKNN\n');
    end
else
    fprintf('Result: The null hyphotesis (H0) cannot be rejected.\n');
end

%Perform a paired t-test comparing simpleKnn with weightedKnn, with a
%significance of 0.05.
fprintf('\nPaired T test (alpha=0.05): CBR(kNN) vs CBR(selectedKNN)\n---------------------------------------------------------\nH0:\tkNN == selectedKNN\nH1:\tkNN != selectedKNN\n');

[simpleVsSelectedH,simpleVsSelectedPValue, simpleVsSelectedCI]  = ttest(kAccuracies, sAccuracies, 0.05, 'both', 2);

fprintf('\nP-value:\t%.4f\n',simpleVsSelectedPValue);
fprintf('Confidence interval:\t[%.4f, %.4f]\n',simpleVsSelectedCI(1),simpleVsSelectedCI(2));

if simpleVsSelectedH == 1
    if simpleVsSelectedCI(2) < 0
        fprintf('Result: H0 rejection! selectedKNN significantly better than kNN\n');
    else
        fprintf('Result: H0 rejection! kNN significantly better than selectedKNN\n');
    end
else
    fprintf('Result: The null hyphotesis (H0) cannot be rejected.\n');
end

%Perform a paired t-test comparing weightedKNN with selectedKNN, with a
%significance of 0.05.
fprintf('\nPaired T test (alpha=0.05): CBR(weightedkNN) vs CBR(selectedKNN)\n---------------------------------------------------------\nH0:\tweightedKNN == selectedKNN\nH1:\tweightedKNN != selectedKNN\n');

[simpleVsSelectedH,simpleVsSelectedPValue, simpleVsSelectedCI]  = ttest(wAccuracies, sAccuracies, 0.05, 'both', 2);

fprintf('\nP-value:\t%.4f\n',simpleVsSelectedPValue);
fprintf('Confidence interval:\t[%.4f, %.4f]\n',simpleVsSelectedCI(1),simpleVsSelectedCI(2));

if simpleVsSelectedH == 1
    if simpleVsSelectedCI(2) < 0
        fprintf('Result: H0 rejection! selectedKNN significantly better than weightedKNN\n');
    else
        fprintf('Result: H0 rejection! weightedKNN significantly better than selectedKNN\n');
    end
else
    fprintf('Result: The null hyphotesis (H0) cannot be rejected.\n');
end


%Wilcoxon signed test
fprintf('\nWILCOXON SIGNED-RANK TEST\n=========================\n');
%Perform a paired t-test comparing simpleKnn with weightedKnn, with a
%significance of 0.05.
fprintf('\nWilcoxon signed-rank test (alpha=0.05): CBR(kNN) vs CBR(weightedkNN)\n------------------------------------------------------------------\nH0:\tkNN == weightedKNN\nH1:\tkNN != weightedKNN\n');

[simVsW_PValue_wilcoxon, simVsW_H_wilcoxon]  = signrank(kAccuracies, wAccuracies, 'alpha', 0.05);

fprintf('\nP-value:\t%.4f\n',simVsW_PValue_wilcoxon);
if simVsW_H_wilcoxon == 1
    fprintf('Result: H0 rejection! There ara a significant difference under Wilcoxon test\n');
else
    fprintf('Result: The null hyphotesis (H0) cannot be rejected.\n');
end

%Perform a paired t-test comparing simpleKnn with selectedKNN, with a
%significance of 0.05.
fprintf('\nWilcoxon signed-rank test: CBR(kNN) vs CBR(selectedKNN)\n---------------------------------------------------------\nH0:\tkNN == selectedKNN\nH1:\tkNN != selectedKNN\n');

[simVsSel_PValue_wilcoxon, simVsSel_H_wilcoxon]  = signrank(kAccuracies, sAccuracies, 'alpha', 0.05);
fprintf('\nP-value:\t%.4f\n',simVsSel_PValue_wilcoxon);
if simVsSel_H_wilcoxon == 1
    fprintf('Result: H0 rejection! There ara a significant difference under Wilcoxon test\n');
else
    fprintf('Result: The null hyphotesis (H0) cannot be rejected.\n');
end

%Perform a paired t-test comparing weightedKnn with selectedKNN, with a
%significance of 0.05.
fprintf('\nWilcoxon signed-rank test: CBR(weightedKNN) vs CBR(selectedKNN)\n-------------------------------------------------------------\nH0:\tweightedKNN == selectedKNN\nH1:\tweightedKNN != selectedKNN\n');

[wVsSel_PValue_wilcoxon, wVsSel_H_wilcoxon]  = signrank(wAccuracies, sAccuracies, 'alpha', 0.05);
fprintf('\nP-value:\t%.4f\n',wVsSel_PValue_wilcoxon);
if wVsSel_H_wilcoxon == 1
    fprintf('Result: H0 rejection! There ara a significant difference under Wilcoxon test\n');
else
    fprintf('Result: The null hyphotesis (H0) cannot be rejected.\n');
end
