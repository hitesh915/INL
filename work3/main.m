dataset = 'breast-w';

data = [];
for fold = 1:10
   [train, test] = parser_nfold(dataset, fold);
   data = [data;{train,test}];
end

train1 = data{1,1};
test1 = data{1,2};

trainData = train1(:,1:end-1);
trainLabels = train1(:,end);
trainLabels(trainLabels == 2) = -1;

testData = test1(:,1:end-1);
testLabels = test1(:,end);
trainLabels(trainLabels == 2) = -1;

svmTrain = train_svm(trainLabels,trainData,1);
predicted = test_svm(svmTrain,testData);