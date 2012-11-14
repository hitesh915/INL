%[features, classes] = parser_arff('data/bal/bal.fold.000000.test.arff');
[trn_feat, trn_clss, tst_feat, tst_clss] = parser_nfold('bal', 1);