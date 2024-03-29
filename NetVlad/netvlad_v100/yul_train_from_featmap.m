function net= yul_train_from_featmap(dbFmTrain, dbFmVal, FmTrain, FmVal, net, info, varargin)
    opts= struct(...
        'netID', 'caffe', ...
        'layerName', 'conv5', ...
        'method', 'vlad_preL2_intra', ...
        'batchSize', 4, ...
        'learningRate', 0.0001, ...
        'lrDownFreq', 8, ...
        'lrDownFactor', 2, ...
        'weightDecay', 0.001, ...
        'momentum', 0.9, ...
        'backPropToLayer', 1, ...
        'fixLayers', [], ...
        'nNegChoice', 1000, ...
        'nNegCap', 10, ...
        'nNegCache', 10, ...
        'nEpoch', 30, ...
        'margin', 0.1, ...
        'excludeVeryHard', false, ...
        'sessionID', [], ...
        'outPrefix', [], ...
        'dbCheckpoint0', [], ...
        'qCheckpoint0', [], ...
        'dbCheckpoint0val', [], ...
        'qCheckpoint0val', [], ...
        'checkpoint0suffix', '', ...
        'info', '', ...
        'test0', true, ...
        'saveFrequency', 2000, ...
        'compFeatsFrequency', 1000, ...
        'computeBatchSize', 10, ...
        'epochTestFrequency', 1, ...
        'doDraw', true, ...
        'printLoss', false, ...
        'printBatchLoss', false, ...
        'nTestSample', 1000, ...
        'nTestRankSample', 5000, ...
        'recallNs', [1:5, 10:5:100], ...
        'useGPU', true, ...
        'numThreads', 12, ...
        'startEpoch', 1, ...
        'clsnum', 101, ...
        'featlen', 64*512, ...
        'net', struct([]) ...
        );
    paths= yul_localPaths();
    opts= vl_argparse(opts, varargin);
    opts.fixLayers = {};
    
    %% ----- Network setup
    if ~isempty(net)
        iepoch_ = net.epoch;
    else
        net= yul_loadNet(opts.netID);
        net.onGPU = 0;
        iepoch_ = 1;
        net.lr = opts.learningRate;
        %% --- Add my layers
        net= yul_addLayers(net, opts, dbFmTrain, FmTrain);

        %% --- Prepare for train
        net= netPrepareForTrain(net);
    end
    
    %% --- BP config
    opts.backPropToLayer= 1;
    opts.backPropToLayerName= net.layers{opts.backPropToLayer}.name;
    opts.backPropDepth= length(net.layers)-opts.backPropToLayer+1;
    assert( all(ismember(opts.fixLayers, relja_layerNames(net))) );
    
    %% --- Train para config
    if opts.useGPU
        net= relja_simplenn_move(net, 'gpu');
    end
    nBatches= floor( dbFmTrain.numVideos / opts.batchSize ); % some might be cut, no biggie
    batchSaveFrequency= ceil(opts.saveFrequency/opts.batchSize);
    progEpoch= tic;
%     lr= opts.learningRate;
    loss_tr = [];
    loss_te = [];
    acc_tr = [];
    acc_te = [];
    res = [];
    lr = opts.learningRate / max(1, opts.lrDownFactor*floor(iepoch_/opts.lrDownFreq));
    %% --- Training
    for iEpoch = iepoch_:opts.nEpoch
        net.epoch = iEpoch;
        relja_progress(iEpoch, opts.nEpoch, 'epoch', progEpoch);
        
        % change learning rate
        if iEpoch~=1 && rem(iEpoch, opts.lrDownFreq)==1
            oldLr= lr;
            lr= lr/opts.lrDownFactor;
            relja_display('Changing learning rate from %f to %f', oldLr, lr); clear oldLr;
        end
        net.lr = lr;
        relja_display('Learning rate %f', lr);
        progBatch= tic;
        if opts.startEpoch>iEpoch, continue; end
        rng(43-1+iEpoch);
        trainOrder= randperm(dbFmTrain.numVideos);
        for iBatch = 1 : nBatches
            relja_progress(iBatch, nBatches, ...
                sprintf('%s epoch %d batch', opts.sessionID, iEpoch), progBatch);
            if rem(iBatch, batchSaveFrequency)==0
                save(sprintf('snapshot/net_%s_iepoch%d_ibatch%d.mat', info, iEpoch, iBatch), 'net');
            end
            bid = trainOrder( (iBatch-1)*opts.batchSize + (1:opts.batchSize) );
            featmap_t = FmTrain(:,:,:,bid);
            class_t = dbFmTrain.label(bid);
            net.layers{end}.class = single(class_t);
            featmap_gpu = gpuArray(featmap_t);
            res= yul_simplenn(net, featmap_gpu, 1, [], ...
                        'backPropDepth', opts.backPropDepth, ... % just for memory
                        'conserveMemoryDepth', true, ...
                        'conserveMemory', false);
            [net,res] = accumulate_gradients(opts, lr, opts.batchSize, net, res) ;
            dzdy = res(end).x/opts.batchSize;
            loss_tr(end+1) = gather(dzdy);
            t = gather(res(end-1).x);
            t = reshape(t, 101, []);
            [~,p] = max(t);
            t = sum(p==class_t');
            acc_tr(end+1) = t / opts.batchSize;
            figure(1)
            plot(loss_tr, 'r');
            hold on
            plot(acc_tr, 'g');
            legend('loss\_tr', 'acc\_tr', 'Location', 'SouthWest');
            hold off
            drawnow;
            %test
            if false
                fprintf('Start testing...');
                loss_t = 0;
                predicted = [];
                testnum = ceil(length(dbFmVal.label)/opts.batchSize);
                tic
                for i_test = 1 : testnum
                    drawnow;
                    fprintf('(%.2f%%)', i_test*100/testnum);
                    testid = mod((i_test-1)*opts.batchSize:i_test*opts.batchSize-1,length(dbFmVal.label))+1;
                    featmap_t = FmVal(:,:,:,testid);
                    class_t = dbFmVal.label(testid);
                    net.layers{end}.class = single(class_t);
                    featmap_gpu = gpuArray(featmap_t);
                    res= yul_simplenn(net, featmap_gpu, 1, [], ...
                                'backPropDepth', opts.backPropDepth, ... % just for memory
                                'conserveMemoryDepth', true, ...
                                'conserveMemory', false);
                    dzdy = res(end).x/opts.batchSize;
                    loss_t = loss_t + gather(dzdy);
                    t = gather(res(end-1).x);
                    t = reshape(t, 101, []);
                    [~,t] = max(t);
                    predicted(testid) = t;
                end
                toc;
                loss_te(end+1) = loss_t / testnum; 
                acc_te(end+1) = sum(predicted==dbFmVal.label') / length(dbFmVal.label);
                figure(2)
                plot(loss_te, 'r');
                fprintf('=====Test loss:%.2f acc:%.2f\n=====', loss_te(end), acc_te(end));
                hold on;
                plot(acc_te, 'g');
                legend('loss\_te', 'acc\_te', 'Location', 'SouthWest');
                hold off;
                drawnow;
                save(sprintf('snapshot/net_%s_acc_%2.2f.mat', info, acc_te(end)*100), 'net');
            end
        end % for ibatch
        
            %Test
            fprintf('Start testing...');
            loss_t = 0;
            predicted = [];
            testnum = ceil(length(dbFmVal.label)/opts.batchSize);
            tic
            for i_test = 1 : testnum
                drawnow;
                fprintf('(%.2f%%)', i_test*100/testnum);
                testid = mod((i_test-1)*opts.batchSize:i_test*opts.batchSize-1,length(dbFmVal.label))+1;
                featmap_t = FmVal(:,:,:,testid);
                class_t = dbFmVal.label(testid);
                net.layers{end}.class = single(class_t);
                featmap_gpu = gpuArray(featmap_t);
                res= yul_simplenn(net, featmap_gpu, 1, [], ...
                            'backPropDepth', opts.backPropDepth, ... % just for memory
                            'conserveMemoryDepth', true, ...
                            'conserveMemory', false);
                dzdy = res(end).x/opts.batchSize;
                loss_t = loss_t + gather(dzdy);
                t = gather(res(end-1).x);
                t = reshape(t, 101, []);
                [~,t] = max(t);
                predicted(testid) = t;
            end
            toc;
            loss_te(end+1) = loss_t / testnum; 
            acc_te(end+1) = sum(predicted==dbFmVal.label') / length(dbFmVal.label);
            figure(2)
            plot(loss_te, 'r');
            fprintf('=====Test loss:%.2f acc:%.2f\n=====', loss_te(end), acc_te(end));
            hold on;
            plot(acc_te, 'g');
            legend('loss\_te', 'acc\_te', 'Location', 'SouthWest');
            hold off;
            drawnow;
            save(sprintf('snapshot/net_%s_acc_%2.2f.mat', info, acc_te(end)*100), 'net');
    end % for iepoch
end

%% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
    predictions = gather(res(end-1).x) ;
    [~,predictions] = sort(predictions, 3, 'descend') ;

    % be resilient to badly formatted labels
    if numel(labels) == size(predictions, 4)
      labels = reshape(labels,1,1,1,[]) ;
    end

    % skip null labels
    mass = single(labels(:,:,1,:) > 0) ;
    if size(labels,3) == 2
      % if there is a second channel in labels, used it as weights
      mass = mass .* labels(:,:,2,:) ;
      labels(:,:,2,:) = [] ;
    end

    error = ~bsxfun(@eq, predictions, labels) ;
    err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
    err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:5,:),[],3)))) ;
end

%% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
    for l=numel(net.layers):-1:1
      for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;

        % accumualte from multiple labs (GPUs) if needed
        if nargin >= 6
          tag = sprintf('l%d_%d',l,j) ;
          tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
          for g = setdiff(1:numel(mmap.Data), labindex)
            tmp = tmp + mmap.Data(g).(tag) ;
          end
          res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
        end


          net.layers{l}.momentum{j} = ...
            opts.momentum * net.layers{l}.momentum{j} ...
            - thisDecay * net.layers{l}.weights{j} ...
            - (1 / batchSize) * res(l).dzdw{j} ;
          net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j} ;

      end
    end
end