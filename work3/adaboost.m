shufflePart = randperm(size(data,1))';

rng(42);
 
data = data(shufflePart, :);
labels = labels(shufflePart, :);

trainData = data(1:300,:);
trainLabels = labels(1:300,:);

testData = data(301:end,:);
testLabels = labels(301:end,:);

trainOurSVM = train_adaboost(trainLabels, trainData, 7);
predicted = test_adaboost(trainOurSVM, testData);
plotAdaboost(trainOurSVM, trainData, trainLabels);
1-sum(predicted~=testLabels)/size(testLabels, 1)