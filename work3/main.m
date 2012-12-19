dataset = 'heart-statlog';

data = [];
 for fold = 1:10
    [train, test] = parser_nfold(dataset, fold);
    data = [data;{train,test}];
 end
 
parametersRBF = cell(64, 2);
parametersLinear = cell(8,1);


cvPartitions = crossValPartition((1:size(data,1))',10,42);

for p = -4:3
    parametersLinear{p+5}{1} = 2^p;
    for pp = -4:3
        parametersRBF{8*p+pp+37}{1} = 2^p;
        parametersRBF{8*p+pp+37}{2} = 2^pp;
    end
end

bestAccuracy = 0;
bestStd = 0;
bestP = cell(1,2);
accuraciesPerParameter = zeros(64,1);
p = 1;
bestPIndex = 1;
while p < 64
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


        svmTrain = train_svm(trainLabels,trainData,parametersRBF{p}{1},parametersRBF{p}{2});
        predicted = test_svm(svmTrain,testData);
        accuracies(k) = 1- sum(predicted~=testLabels)/size(testLabels,1);
    end
    
    fprintf('sigma:\t%f, C:\t%f, Accuracy:\t%f\n',parametersRBF{p}{2}, parametersRBF{p}{1}, mean(accuracies));
    
    if bestAccuracy <= mean(accuracies)
        bestAccuracy = mean(accuracies);
        bestStd = std(accuracies);
        bestP{1}{1} = parametersRBF{p}{1};
        bestP{1}{2} = parametersRBF{p}{2};
        bestPIndex = p;
    else
        if bestPIndex == p - 1
            cI = floor(p/8)+1;
            p = cI*8;
        end
    end
    p = p+1;
    p
end

bestAccuracy
