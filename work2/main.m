%[features, classes] = parser_arff('data/bal/bal.fold.000000.test.arff');
[trn_matrix, tst_matrix] = parser_nfold('bal', 1);