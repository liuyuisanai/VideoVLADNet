% paths_video = yul_localPaths();
% db_video = yul_get_ucf101(paths_video, 'trainlist01.txt');
% dirs = arrayfun(@(i_t)dir(fullfile(db_video.list{i_t}, '*.jpg')), 1:length(db_video.list), 'UniformOutput', false);
gpunum = 8;
dirs = arrayfun(@(i_t)dir(fullfile(db.list{i_t}, '*.jpg')), 1:length(db.list), 'UniformOutput', false);
parfor i_gpu = 1 : gpunum
    gputic = tic();
    gpuDevice(i_gpu);
    net= yul_loadNet('vd16', 'pool5');
    net= relja_simplenn_move(net, 'gpu');
    ids{i_gpu} = find(mod(1 : length(db.list), gpunum)+1 == i_gpu);
    ids_t = ids{i_gpu};
    for i = 1 : length(ids_t)
        tic
        thisid = ids_t(i);
        fprintf('Solver %d: Loading feature for %d/%d...', i_gpu, i, length(ids_t));
        list = arrayfun(@(i_t)fullfile(db.list{ids_t(i)}, dirs{ids_t(i)}(i_t).name), 1:length(dirs{ids_t(i)}), 'UniformOutput', false);
        ims_ = vl_imreadjpeg(list, 'numThreads', 16);
        for iIm= 1:length(ims_)
            if size(ims_{iIm},3)==1
                ims_{iIm}= cat(3,ims_{iIm},ims_{iIm},ims_{iIm});
            end
        end
        ims= cat(4, ims_{:});

        ims(:,:,1,:)= ims(:,:,1,:) - net.normalization.averageImage(1,1,1);
        ims(:,:,2,:)= ims(:,:,2,:) - net.normalization.averageImage(1,1,2);
        ims(:,:,3,:)= ims(:,:,3,:) - net.normalization.averageImage(1,1,3);
        img_t = gpuArray(ims);
        res= relja_simplenn(net, img_t, [], [], ...
        'backPropDepth', 0, ... % just for memory
        'conserveMemoryDepth', true, ...
        'conserveMemory', false);
%         featmap{i_gpu}{i} = gather(yul_put_n_2_h(res(end).x));
        filename = sprintf('D:/YuLiu/video/feature_pool5/vgg16_conv5_3_ucf101_train01_%d.bin', thisid);
        writebin(gather(yul_put_n_2_h(res(end).x)), filename);
        toc
    end
    fprintf('\n\nSolver %d:Done.', i_gpu);
    toc(gputic)
end