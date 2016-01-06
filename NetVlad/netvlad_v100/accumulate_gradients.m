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