%% Config
setup;
netID= 'featmapVlad';
paths= yul_localPaths();
dbTrain= yul_get_ucf101(paths, 'trainlist01.txt');
dbVal= yul_get_ucf101(paths, 'testlist01.txt');
lr= 0.01;
net=[];

%% Cannot modified
dbFm_train = yul_get_dbFm(dbTrain, paths);
dbFm_test = yul_get_dbFm(dbVal, paths);
dbFm_test.numVideos = 3779;
dbFm_test.path(2605:2608)=[];
dbFm_test.label(2605:2608)=[];
%read featmap(needs large time)
trainfeatmap = yul_read_featmap_from_bin(dbFm_train.path, [112, 10, 512]);
testfeatmap = yul_read_featmap_from_bin(dbFm_test.path, [112, 10, 512]);
load('snapshot/net0_pool5.mat');
load('initdata\softmax_weight.mat');
net.layers{5}.weights{1}(1,1,:,:) = gpuArray(weight);
net.layers{5}.weightDecay(:) = 1e-12;
net.layers{5}.learningRate(:) = 1;
sessionID= yul_train_from_featmap(dbFm_train, dbFm_test, trainfeatmap, testfeatmap, net, '(100_1_64)', ...
    'netID', netID, ...
    'method', 'vlad_preL2_intra', ...
    'learningRate', lr, ...
    'doDraw', true, ...
    'batchSize', 64, ...
    'nEpoch', 50, ...
    'layerName', 'pool5' ...
    );