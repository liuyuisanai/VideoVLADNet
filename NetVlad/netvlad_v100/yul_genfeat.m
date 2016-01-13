function feature = yul_genfeat(dbFm, net, featlen, batchsize, gpuid)
    
    net= relja_simplenn_move(net, 'gpu');
    net.layers(end-1:end)=[];
    if nargin < 3
        batchsize = 32;
        gpuid = 1;
    end
    gpunum = numel(gpuid);
    num = length(dbFm.path);
    feature = zeros(num, featlen, 'single');
    if length(gpuid) < 2
        for i = 1 : ceil(num/batchsize)
            id_batch = mod((i-1)*batchsize:i*batchsize-1, num)+1;
            featmap_t = gpuArray(yul_read_featmap_from_bin(dbFm.path(id_batch), [112, 10, 512]));
            res = yul_simplenn(net, gpuArray(featmap_t));
            feature(id_batch,:) = reshape(gather(res(end).x), [featlen, batchsize])';
        end
    else
        if matlabpool('size') ~= gpunum
            matlabpool(gpunum);
        end
        net_cpu = relja_simplenn_move(net, 'cpu');
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
                featmap_t = gpuArray(yul_read_featmap_from_bin(dbFm.path(id_batch), [112, 10, 512]));
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