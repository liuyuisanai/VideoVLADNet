preL2 = false;

if preL2
    meanpool_tr_preL2 = zeros(8*10*512, 9537, 'single');
    meanpool_te_preL2 = zeros(8*10*512, 3779, 'single');
else
    meanpool_tr = zeros(8*10*512, 9537, 'single');
    meanpool_te = zeros(8*10*512, 3779, 'single');
end


parfor i = 1 : size(meanpool_tr, 2)
    if preL2
        featmap_t = reshape(vl_nnnormalize(vgg19pool5_caffe_train(:,:,:,:,i), [1024,0,1,0.5]), [8*10, 512, 16]);
        featmap_t = permute(featmap_t, [2 1 3]);
        featmap_t = reshape(featmap_t, [], 16);
        meanpool_tr_preL2(:,i) = mean(featmap_t, 2);
    else
        featmap_t = reshape(vgg19pool5_caffe_train(:,:,:,:,i), [8*10, 512, 16]);
        featmap_t = permute(featmap_t, [2 1 3]);
        featmap_t = reshape(featmap_t, [], 16);
        meanpool_tr(:,i) = mean(featmap_t, 2);
    end
end

parfor i = 1 : size(meanpool_te, 2)
    if preL2
        featmap_t = reshape(vl_nnnormalize(vgg19pool5_caffe_test(:,:,:,:,i), [1024,0,1,0.5]), [8*10, 512, 16]);
        featmap_t = permute(featmap_t, [2 1 3]);
        featmap_t = reshape(featmap_t, [], 16);
        meanpool_te_preL2(:,i) = mean(featmap_t, 2);
    else
        featmap_t = reshape(vgg19pool5_caffe_test(:,:,:,:,i), [8*10, 512, 16]);
        featmap_t = permute(featmap_t, [2 1 3]);
        featmap_t = reshape(featmap_t, [], 16);
        meanpool_te(:,i) = mean(featmap_t, 2);
    end
end

% model = mpsvm.train(label_train, sparse(double(meanpool_tr_norm)), '-s 0 -n 32 -c 10 -q', 'col');
% [predicted_label] = mpsvm.predict(label_test, sparse(double(meanpool_te_norm)), model,'', 'col');

model = mpsvm.train(label_train, sparse(double(meanpool_tr)), '-s 0 -n 32 -c 0.1 -q', 'col');
fprintf('MeanPool:')
[predicted_label] = mpsvm.predict(label_test, sparse(double(meanpool_te)), model,'', 'col');

if false
    model = mpsvm.train(label_train, sparse(double(normc(meanpool_tr))), '-s 0 -n 32 -c 0.1 -q', 'col');
    fprintf('MeanPool:')
    [predicted_label] = mpsvm.predict(label_test, sparse(double(normc(meanpool_te))), model,'', 'col');
end