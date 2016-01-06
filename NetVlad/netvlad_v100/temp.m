% paths_video = yul_localPaths();
% db_video = yul_get_ucf101(paths_video, 'trainlist01.txt');
% dirs = arrayfun(@(i_t)dir(fullfile(db_video.list{i_t}, '*.jpg')), 1:length(db_video.list), 'UniformOutput', false);
    gputic = tic();
    net= loadNet('vd16', 'conv5_3');
    net= relja_simplenn_move(net, 'gpu');
    ids_t = 1:numel(dbVal.list);
%     dirs = arrayfun(@(i_t)dir(fullfile(dbVal.list{i_t}, '*.jpg')), 1:length(dbVal.list), 'UniformOutput', false);
    for i = 1 : length(ids_t)
        tic
        thisid = ids_t(i);
        fprintf('Solver %d: Loading feature for %d/%d...', 1, i, length(ids_t));
        list = arrayfun(@(i_t)fullfile(dbVal.list{ids_t(i)}, dirs{ids_t(i)}(i_t).name), 1:length(dirs{ids_t(i)}), 'UniformOutput', false);
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
        filename = sprintf('G:/temp/video/feature/vgg16_conv5_3_ucf101_test01_%d.bin', thisid);
        writebin(gather(yul_put_n_2_h(res(end).x)), filename);
        toc
    end
    fprintf('\n\nSolver %d:Done.', i_gpu);
    toc(gputic)
