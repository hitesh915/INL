% shufflePart = randperm(size(data,1))';
% 
% rng(42);
%  
% data = data(shufflePart, :);
% labels = labels(shufflePart, :);
% 
% trainData = data(1:70,:);
% trainLabels = labels(1:70,:);
% 
% testData = data(71:end,:);
% testLabels = labels(71:end,:);

trainOurSVM = train_adaboost(trainLabels, trainData, 50,'svm');
predicted = test_adaboost(testData,trainOurSVM);
%plotAdaboost(trainOurSVM, trainData, trainLabels);
1-sum(predicted~=testLabels)/size(testLabels, 1)