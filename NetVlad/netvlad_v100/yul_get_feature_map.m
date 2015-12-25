function featmap = yul_get_feature_map(db)
    net= loadNet('vd16', 'conv5_3');
    dirs = arrayfun(@(i_t)dir(fullfile(db.list{i_t}, '*.jpg')), 1:length(db.list), 'UniformOutput', false);
    imageFns = arrayfun(@(j_t)arrayfun(@(i_t)fullfile(db.list{j_t}, dirs{j_t}(i_t).name), 1:length(dirs{j_t}), 'UniformOutput', false), 1:length(db.list), 'UniformOutput', false);
    thisNumIms= length(imageFns);
    ims_= arrayfun(@(i_t)vl_imreadjpeg(imageFns{i_t}, 'numThreads', 16), 1:length(db.list), 'UniformOutput', false);
    for i = 1 : length(db.list)
        img_t = gpuArray(ims{i});
        res= relja_simplenn(net, img_t, [], [], ...
                        'backPropDepth', 0, ... % just for memory
                        'conserveMemoryDepth', true, ...
                        'conserveMemory', false);
        featmap{i} = yul_put_n_2_h(res(end).x);
    end
end