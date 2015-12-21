function paths= localPaths()
    
    % --- dependencies
    
    % refer to README.md for the information on dependencies
    
    paths.libReljaMatlab= 'D:/YuLiu/place/NetVlad/netvlad_v100/relja_matlab_v100/';
    paths.libMatConvNet= 'D:/YuLiu/place/NetVlad/netvlad_v100/matconvnet-1.0-beta16_gpu/'; % should contain matlab/
    
    % If you have installed yael_matlab (**highly recommended for speed**),
    % provide the path below. Otherwise, provide the path as 'yael_dummy/':
    % this folder contains my substitutes for the used yael functions,
    % which are **much slower**, and only included for demonstration purposes
    % so do consider installing yael_matlab, or make your own faster
    % version (especially of the yael_nn function)
    paths.libYaelMatlab= 'yael_dummy/';
    
    % --- dataset specifications
    
    paths.dsetSpecDir= 'D:/YuLiu/place/NetVlad/netvlad_v100/datasets/';
    
    % --- dataset locations
    paths.dsetRootPitts= ''; % should contain images/ and queries/
    paths.dsetRootTokyo247= 'D:/YuLiu/place/dataset/tokyo247/'; % should contain images/ and query/
    paths.dsetRootTokyoTM= 'D:/YuLiu/place/dataset/tokyoTimeMachine/'; % should contain images/
    
    % --- our networks
    % models used in our paper, download them from our research page
    paths.ourCNNs= 'D:/YuLiu/place/NetVlad/netvlad_v100/models/';
    
    % --- pretrained networks
    % off-the-shelf networks trained on other tasks, available from the MatConvNet
    % website: http://www.vlfeat.org/matconvnet/pretrained/
    paths.pretrainedCNNs= '';
    
    % --- initialization data (off-the-shelf descriptors, clusters)
    % Not necessary: these can be computed automatically, but it is recommended
    % in order to use the same initialization as we used in our work
    paths.initData= 'D:/YuLiu/place/NetVlad/netvlad_v100/initdata/';
    
    % --- output directory
    paths.outPrefix= 'D:/YuLiu/place/NetVlad/netvlad_v100/output/';
end
