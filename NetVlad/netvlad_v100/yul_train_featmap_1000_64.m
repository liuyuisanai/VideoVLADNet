%% Config
setup;
netID= 'vgg19';
layerName = 'pool5';
paths= yul_localPaths();
dbTrain= yul_get_ucf101(paths, 'trainlist01.txt');
dbVal= yul_get_ucf101(paths, 'testlist01.txt');
lr= 0.001;
net=[];

%% Cannot modified
dbFm_train = yul_get_dbFm(dbTrain, paths);
dbFm_test = yul_get_dbFm(dbVal, paths);
% dbFm_test.numVideos = 3779;
% dbFm_test.path(2605:2608)=[];
% dbFm_test.label(2605:2608)=[];
%read featmap(needs large time)
if ~exist('trainfeatmap', 'var')
%     trainfeatmap = yul_read_featmap_from_bin(dbFm_train.path, [112, 10, 512]);
%     testfeatmap = yul_read_featmap_from_bin(dbFm_test.path, [112, 10, 512]);
    load('datasets\vgg19pool5_chuang.mat');
end
load('snapshot/net0_vgg19_pool5.mat');
if false
    load('initdata\softmax_weight.mat');
    net.layers{5}.weights{1}(1,1,:,:) = gpuArray(weight);
    net.layers{5}.weightDecay(:) = 1e-12;
    net.layers{5}.learningRate(:) = 1;
end
sessionID= yul_train_from_featmap(dbFm_train, dbFm_test, trainfeatmap, testfeatmap, net, '(100_1_64)', ...
    'netID', netID, ...
    'method', 'vlad_preL2_intra', ...
    'learningRate', lr, ...
    'doDraw', true, ...
    'batchSize', 64, ...
    'nEpoch', 50, ...
    'layerName', 'pool5' ...
    );