function clsts= yul_getClusters(net, opts, clstFn, k, dbFm, trainDescFn)
    
    if ~exist(clstFn, 'file')
        
        if ~exist(trainDescFn, 'file')
            
            if opts.useGPU
                net= relja_simplenn_move(net, 'gpu');
            end
            
            % ---------- extract training descriptors
            
            relja_display('Computing training descriptors');
            
            nTrain= 256000;
            nPerImage= 30;
            nIm= ceil(nTrain/nPerImage);
            
            rng(43);
            trainIDs= randsample(dbFm.numVideos, nIm);
            
            nTotal= 0;
            
            prog= tic;
            
            for iIm= 1:nIm
                relja_progress(iIm, nIm, 'extract train descs', prog);
                
                % --- extract descriptors
                
                % didn't want to complicate with batches here as it's only done once (per network and training set)
                
                fm= readbin(dbFm.path{iIm}, [112, 10, 512], 'single');
                
                if opts.useGPU
                    fm= gpuArray(fm);
                end
                
                res= vl_simplenn(net, fm, [], [], 'conserveMemory', true);
                descs= gather(res(end).x);
                descs= reshape( descs, [], size(descs,3) )';
                
                % --- sample descriptors
                
                nThis= min( min(nPerImage, size(descs,2)), nTrain - nTotal );
                descs= descs(:, randsample( size(descs,2), nThis ) );
                
                if iIm==1
                    trainDescs= zeros( size(descs,1), nTrain, 'single' );
                end
                
                trainDescs(:, nTotal+[1:nThis])= descs;
                nTotal= nTotal+nThis;
            end
            
            trainDescs= trainDescs(:, 1:nTotal);
            
            % move back to CPU addLayers() assumes it
            if opts.useGPU
                net= relja_simplenn_move(net, 'cpu');
            end
            
            save(trainDescFn, 'trainDescs');
        else
            relja_display('Loading training descriptors');
            load(trainDescFn, 'trainDescs');
        end
        
        % ---------- Cluster descriptors
        
        relja_display('Computing clusters');
        clsts= yael_kmeans(trainDescs, k, 'niter', 500, 'verbose', 2, 'seed', 43);
        clear trainDescs;
        
        save(clstFn, 'clsts');
    else
        relja_display('Loading clusters');
        load(clstFn, 'clsts');
        assert(size(clsts, 2)==k);
    end
    
end
