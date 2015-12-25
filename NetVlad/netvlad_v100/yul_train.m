setup;
netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white';
paths= yul_localPaths();
dbTrain= yul_get_ucf101(paths, 'trainlist01.txt');
dbVal= yul_get_ucf101(paths, 'testlist01.txt');
lr= 0.001;
sessionID= yul_train_softmax(dbTrain, dbVal, ...
    'netID', 'vd16', 'layerName', 'conv5_3', 'backPropToLayer', 'conv5_3', ...
    'method', 'vlad_preL2_n2h_intra', ...
    'learningRate', lr, ...
    'doDraw', true);