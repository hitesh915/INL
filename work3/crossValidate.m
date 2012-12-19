function [ eout ] = crossValidate( dataset )
%CROSSVALIDATE Summary of this function goes here
%   Detailed explanation goes here

    % ----------------------------------------------------------------
    % -- Data management functions
    % ----------------------------------------------------------------

    % Parse data folds
    function [folds] = parseFolds( dataset )
        folds = [];
        for fold = 1:10
           [~, fdata] = parser_nfold(dataset, fold);
           tfold = struct('data', fdata(:,1:end-1), 'labels', fdata(:,end));
           tfold.labels(tfold.labels == 2) = -1;
           folds = [folds ; tfold];
        end
    end

    % Function to merge folds, turns N folds in num folds
    function [mfolds] = mergeFolds( folds, num )
        cfolds = size(folds,1);
        
        % Initialize merged folds
        mfolds = [];
        for nf = 1:num
            mfolds = [mfolds ; struct('data', [], 'labels', [])];
        end
        
        % Assign folds to merged folds
        for of = 1:cfolds
            df = mod(of-1, num) + 1;
            mfolds(df).data   = [mfolds(df).data   ; folds(of).data];
            mfolds(df).labels = [mfolds(df).labels ; folds(of).labels];
        end
    end

    % ----------------------------------------------------------------
    % -- Model optimization functions
    % ----------------------------------------------------------------

    % Initialize parameter arrays
    parametersRBF      = zeros(64, 2);
    parametersLinear   = zeros(8,1);
    parametersAdaboost = zeros(8,1);

    for p = -4:3
        parametersLinear(p+5, 1) = 2^p;
        parametersAdaboost(p+5, 1) = 3*(p+5);
        for pp = -4:3
            parametersRBF(1+8*(p+4)+(pp+4), 1) = 2^p;
            parametersRBF(1+8*(p+4)+(pp+4), 2) = 2^pp;
        end
    end
    
    function [model, errs] = optimizeLinearSVM(folds)
        nfolds = size(folds,1);

        models = cell(8,1);
        errs = zeros(8,1);
        
        for it = 1:nfolds
            train = mergeFolds(folds(1:nfolds ~= it), 1);
            test  = folds(it);

            for ic = 1:8
                c = parametersLinear(ic);
                
                models{ic} = train_svm(train.labels, train.data, c);
                tlabels = test_svm(test.data, models{ic});
                errs(ic) = errs(ic) + sum(test.labels ~= tlabels) / size(tlabels,1);
            end
        end
        
        % Get final errors
        errs = errs / nfolds;
        
        % Select best model
        [min_error, ind] = min(errs);
        model = models{ind};
        
        % Print result
        for ic = 1:8
            c = parametersLinear(ic);
            fprintf('  SVM (c = %.4f) error: %.4f\n', c, errs(ic));
        end
    end

    function [model, errs] = optimizeRbfSVM(folds)
        nfolds = size(folds,1);

        models = cell(64,1);
        errs = zeros(64,1);
        
        fprintf('  Wrapping space and time... ');
        
        psize = 0;
        for it = 1:nfolds
            train = mergeFolds(folds(1:nfolds ~= it), 1);
            test  = folds(it);

            for ico = 1:64
                c = parametersRBF(ico,1);
                o = parametersRBF(ico,2);

                msg = sprintf('%.2f pct.', ((it - 1) * 64 + ico) * 100 / (64*3));
                fprintf(repmat('\b', 1, psize));
                fprintf('%s', msg);
                psize = numel(msg);
                
                models{ico} = train_svm(train.labels, train.data, c, o);
                tlabels = test_svm(test.data, models{ico});
                errs(ico) = errs(ico) + sum(test.labels ~= tlabels) / size(tlabels,1);
            end
        end
        
        % Get final errors
        errs = errs / nfolds;
        
        % Select best model
        [min_error, ind] = min(errs);
        model = models{ind};
        
        % Print result
        for ico = 1:64
            c = parametersRBF(ico,1);
            o = parametersRBF(ico,2);
            fprintf('  RBF (c = %.4f, o = %.6f) error: %.4f\n', c, o, errs(ico));
        end
    end

    function [model, errs] = optimizeAdaboost(folds)
        nfolds = size(folds,1);

        models = cell(8,1);
        errs = zeros(8,1);
        
        for it = 1:nfolds
            train = mergeFolds(folds(1:nfolds ~= it), 1);
            test  = folds(it);

            for iv = 1:8
                t = parametersAdaboost(iv);
                
                models{iv} = train_adaboost(train.labels, train.data, t);
                tlabels = test_adaboost(test.data, models{iv});
                errs(iv) = errs(iv) + sum(test.labels ~= tlabels) / size(tlabels,1);
            end
        end
        
        % Get final errors
        errs = errs / nfolds;
        
        % Select best model
        [min_error, ind] = min(errs);
        model = models{ind};
        
        % Print result
        for iv = 1:8
            t = parametersAdaboost(iv);
            fprintf('  ADA (t = %.4f) error: %.4f\n', t, errs(iv));
        end
    end

    % ----------------------------------------------------------------
    % -- General algorithm
    % ----------------------------------------------------------------

    % Parse data folds
    data = parseFolds(dataset);

    eout_svm = zeros(size(data,1));
    eout_rbf = zeros(size(data,1));
    eout_ada = zeros(size(data,1));
    
    errs_svm = zeros(8,  1);
    errs_rbf = zeros(64, 1);
    errs_ada = zeros(8,  1);
    
    % Iterate through the 10-fold cross-validation
    for i = 1:size(data,1)
        threeFolds = mergeFolds(data(1:size(data,1) ~= i), 3);
        trainData  = mergeFolds(data(1:size(data,1) ~= i), 1);
        testData = data(i);
        
        % Select optimal parameter values 3-fold way
        fprintf('10-fold cross-validation, fold %s:\n', int2str(i));
        [svm_model, svm_errs] = optimizeLinearSVM(threeFolds);
        [rbf_model, rbf_errs] = optimizeRbfSVM(threeFolds);
        [ada_model, ada_errs] = optimizeAdaboost(threeFolds);
        
        % Add error surfaces
        errs_svm = errs_svm + svm_errs;
        errs_rbf = errs_rbf + rbf_errs;
        errs_ada = errs_ada + ada_errs;
        
        % SVM: Calculate out of sample error
        svm_model = train_svm(trainData.labels, trainData.data, svm_model.c);
        ilabels = test_svm(testData.data, svm_model);
        eout_svm(i) = eout_svm(i) + sum(testData.labels ~= ilabels) / size(ilabels,1);
        
        % RBF: Calculate out of sample error
        rbf_model = train_svm(trainData.labels, trainData.data, rbf_model.c, rbf_model.sigma);
        ilabels = test_svm(testData.data, rbf_model);
        eout_rbf(i) = eout_rbf(i) + sum(testData.labels ~= ilabels) / size(ilabels,1);
    
        % ADA: Calculate out of sample error
        ada_model = train_adaboost(trainData.labels, trainData.data, ada_model.t);
        ilabels = test_adaboost(testData.data, ada_model);
        eout_ada(i) = eout_ada(i) + sum(testData.labels ~= ilabels) / size(ilabels,1);
    end
    
    % Calculate mean eout errors
    meout_svm = sum(eout_svm) / size(eut_svm,1);
    meout_rbf = sum(eout_rbf) / size(eut_rbf,1);
    meout_ada = sum(eout_ada) / size(eut_ada,1);
    
    % Prepare eout return structure
    eout = struct;
    eout.svm = struct;
    eout.svm.mean = meout_svm;
    eout.svm.surface = errs_svm;
    eout.rbf = struct;
    eout.rbf.mean = meout_rbf;
    eout.rbf.surface = errs_rbf;
    eout.ada = struct;
    eout.ada.mean = meout_ada;
    eout.ada.surface = errs_ada;
end

