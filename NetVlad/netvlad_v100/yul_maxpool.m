preL2 = false;

if preL2
    maxpool_tr_preL2 = zeros(8*10*512, 9537, 'single');
    maxpool_te_preL2 = zeros(8*10*512, 3779, 'single');
else
    maxpool_tr = zeros(8*10*512, 9537, 'single');
    maxpool_te = zeros(8*10*512, 3779, 'single');
end


parfor i = 1 : size(maxpool_tr, 2)
    if preL2
        featmap_t = reshape(vl_nnnormalize(vgg19pool5_caffe_train(:,:,:,:,i), [1024,0,1,0.5]), [8*10, 512, 16]);
        featmap_t = permute(featmap_t, [2 1 3]);
        featmap_t = reshape(featmap_t, [], 16);
        maxpool_tr_preL2(:,i) = max(featmap_t, [], 2);
    else
        featmap_t = reshape(vgg19pool5_caffe_train(:,:,:,:,i), [8*10, 512, 16]);
        featmap_t = permute(featmap_t, [2 1 3]);
        featmap_t = reshape(featmap_t, [], 16);
        maxpool_tr(:,i) = max(featmap_t, [], 2);
    end
end

parfor i = 1 : size(maxpool_te, 2)
    if preL2
        featmap_t = reshape(vl_nnnormalize(vgg19pool5_caffe_test(:,:,:,:,i), [1024,0,1,0.5]), [8*10, 512, 16]);
        featmap_t = permute(featmap_t, [2 1 3]);
        featmap_t = reshape(featmap_t, [], 16);
        maxpool_te_preL2(:,i) = max(featmap_t, [], 2);
    else
        featmap_t = reshape(vgg19pool5_caffe_test(:,:,:,:,i), [8*10, 512, 16]);
        featmap_t = permute(featmap_t, [2 1 3]);
        featmap_t = reshape(featmap_t, [], 16);
        maxpool_te(:,i) = max(featmap_t, [], 2);
    end
end

model = mpsvm.train(label_train, sparse(double(maxpool_tr)), '-s 0 -n 32 -c 10 -q', 'col');
[predicted_label] = mpsvm.predict(label_test, sparse(double(maxpool_te)), model,'', 'col');

% model = mpsvm.train(label_train, sparse(double(maxpool_tr_preL2)), '-s 0 -n 32 -c 10 -q', 'col');
% [predicted_label] = mpsvm.predict(label_test, sparse(double(maxpool_te_preL2)), model,'', 'col');