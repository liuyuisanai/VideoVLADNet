%% Config
setup;
netID= 'featmapVlad';
paths= yul_localPaths();
dbTrain= yul_get_ucf101(paths, 'trainlist01.txt');
dbVal= yul_get_ucf101(paths, 'testlist01.txt');
lr= 0.001;

%% Cannot modified
dbFm_train = yul_get_dbFm(db_video_train, paths);
dbFm_test = yul_get_dbFm(db_video_test, paths);
sessionID= yul_train_from_featmap(dbFm_train, dbFm_test, ...
    'netID', netID, ...
    'method', 'vlad_preL2_intra', ...
    'learningRate', lr, ...
    'doDraw', true, ...
    'batchSize', 16 ...
    );