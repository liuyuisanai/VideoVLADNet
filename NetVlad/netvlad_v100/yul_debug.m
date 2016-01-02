for i = 1 : 100
    res= yul_simplenn(net, featmap_gpu, 1, res, ...
                            'backPropDepth', opts.backPropDepth, ... % just for memory
                            'conserveMemoryDepth', true, ...
                            'conserveMemory', false);
    [net,res] = accumulate_gradients(opts, lr, opts.batchSize, net, res) ;
    dzdy = res(end).x;
    fprintf('loss=%f\n', dzdy/opts.batchSize);
end