function feature = yul_genfeat(dbFm, net, featlen, batchsize)
    net= relja_simplenn_move(net, 'gpu');
    net.layers(end-1:end)=[];
    if nargin < 3
        batchsize = 32;
    end
    num = length(dbFm.path);
    feature = zeros(num, featlen, 'single');
    for i = 1 : ceil(num/batchsize)
        id_batch = mod((i-1)*batchsize:i*batchsize-1, num)+1;
        featmap_t = gpuArray(yul_read_featmap_from_bin(dbFm.path(id_batch), [240, 20, 512]));
        res = yul_simplenn(net, gpuArray(featmap_t));
        feature(id_batch,:) = reshape(gather(res(end).x), [featlen, batchsize])';
    end
end