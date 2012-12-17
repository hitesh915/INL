shufflePart = randperm(size(data,1))';
%  
rng(42);
 
data = data(shufflePart, :);
labels = labels(shufflePart, :);

trainData = data(1:380,:);
trainLabels = labels(1:380,:);

testData = data(381:end,:);
testLabels = labels(381:end,:);
trainOurSVM = train_svm(trainLabels, trainData, 0.1,100);
[hAxis,hLines] = svmplotdata(trainData,trainLabels,trainOurSVM);

predicted = test_svm(trainOurSVM, testData);

% trainSVM = svmtrain(trainData, trainLabels, 'method', 'QP','showplot',true,'boxconstraint',1);
% predicted = svmclassify(trainSVM, testData);

% predicted
% 
1-sum(predicted~=testLabels)/size(testLabels, 1)