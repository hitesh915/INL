shufflePart = randperm(size(data,1))';

rng(42);

data = data(shufflePart, :);
labels = labels(shufflePart, :);

trainData = data(1:80,:);
trainLabels = labels(1:80,:);

testData = data(81:end,:);
testLabels = labels(81:end,:);

trainOurSVM = train_svm(trainLabels, trainData, 1, 1);
[hAxis,hLines] = svmplotdata(trainData,trainLabels,trainOurSVM);

predicted = test_svm(trainOurSVM, testData);

%trainSVM = svmtrain(trainData, trainLabels, 'method', 'QP','kernel_function','rbf','showplot',true,'boxconstraint',1,'rbf_sigma',1);
% predicted = svmclassify(trainSVM, testData);
% 
% predicted
% 
1-sum(predicted~=testLabels)/size(testLabels, 1)