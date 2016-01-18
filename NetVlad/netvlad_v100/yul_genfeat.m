function feature = yul_genfeat(fm, net, featlen, batchsize, gpuid)
    
    net= relja_simplenn_move(net, 'gpu');
    net.layers(end-1:end)=[];
    if nargin < 3
        batchsize = 32;
        gpuid = 1;
    end
    net_cpu = relja_simplenn_move(net, 'cpu');
    gpunum = numel(gpuid);
    num = size(fm, 4);
    feature = zeros(num, featlen, 'single');
    if length(gpuid) < 2
        gpuDevice(gpuid);
        net_gpu = relja_simplenn_move(net_cpu, 'gpu');
        for i = 1 : ceil(num/batchsize)
            tic
            fprintf('Cluster %d -- %d/%d ', 1, i, ceil(num/batchsize));
            id_batch = mod((i-1)*batchsize:i*batchsize-1, num)+1;
            featmap_t = gpuArray(fm(:,:,:,id_batch));
            res = yul_simplenn(net_gpu, gpuArray(featmap_t));
            feature(id_batch,:) = reshape(gather(res(end).x), [featlen, batchsize])';
            toc
        end
    else
        if matlabpool('size') ~= gpunum
            matlabpool(gpunum);
        end
        for i = 1 : gpunum
            ids{i} = mod((i-1)*ceil(num/gpunum):i*ceil(num/gpunum)-1, num)+1;
        end
        parfor i = 1 : gpunum
            gpuDevice(i);
            net_gpu = relja_simplenn_move(net_cpu, 'gpu');
            num_t = numel(ids{i});
            feature_t{i} = zeros(num_t, featlen, 'single');
            for i_t = 1 : ceil(num_t/batchsize)
                if mod(i_t, 5) == 1
                    fprintf('Cluster %d -- %d/%d\n', i, i_t, ceil(num_t/batchsize));
                end
                id_batch = ids{i}(mod((i_t-1)*batchsize:i_t*batchsize-1, num_t)+1);
                featmap_t = gpuArray(fm(:,:,:,id_batch));
                res = yul_simplenn(net_gpu, gpuArray(featmap_t));
                feature_t{i}(mod((i_t-1)*batchsize:i_t*batchsize-1, num_t)+1,:) =...
                    reshape(gather(res(end).x), [featlen, batchsize])';
            end
        end
        for i = 1 : gpunum
            feature(ids{i},:) = feature_t{i};
        end
    end
end