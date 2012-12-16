dataset = 'ionosphere';

data = [];
 for fold = 1:10
    [train, test] = parser_nfold(dataset, fold);
    data = [data;{train,test}];
 end
 
accuracies = zeros(10,1);

for k = 1:10
    train = data{k,1};
    test = data{k,2};
    
    trainData = train(:,1:end-1);
    trainLabels = train(:,end);
    trainLabels(trainLabels == 2) = -1;

    testData = test(:,1:end-1);
    testLabels = test(:,end);
    testLabels(testLabels == 2) = -1;
    svmTrain = train_svm(trainLabels,trainData,0.5,4);
    predicted = test_svm(svmTrain,testData);
    accuracies(k) = 1- sum(predicted~=testLabels)/size(testLabels,1)
end

mean(accuracies)
std(accuracies)