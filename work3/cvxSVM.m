shufflePart = randperm(size(data,1))';
%  
rng(42);
 
data = data(shufflePart, :);
labels = labels(shufflePart, :);

trainData = data(1:120,:);
trainLabels = labels(1:120,:);

testData = data(121:end,:);
testLabels = labels(121:end,:);
trainOurSVM = train_svm(trainLabels, trainData, 2,2);
[hAxis,hLines] = svmplotdata(trainData,trainLabels,trainOurSVM);

predicted = test_svm(trainOurSVM, testData);

% trainSVM = svmtrain(trainData, trainLabels, 'method', 'QP','showplot',true,'boxconstraint',1);
% predicted = svmclassify(trainSVM, testData);

% predicted
% 
1-sum(predicted~=testLabels)/size(testLabels, 1)