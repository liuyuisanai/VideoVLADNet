preL2 = true;

if preL2
    maxpool_tr_preL2 = zeros(7*10*512, 9537, 'single');
    maxpool_te_preL2 = zeros(7*10*512, 3779, 'single');
else
    maxpool_tr = zeros(7*10*512, 9537, 'single');
    maxpool_te = zeros(7*10*512, 3779, 'single');
end


parfor i = 1 : size(avepool_tr, 2)
    featmap_t = reshape(vggpool5_tr(:,:,:,:,i), [7*10, 512, 16]);
    featmap_t = permute(featmap_t, [2 1 3]);
    if preL2
        for j = 1 : 16
            featmap_t(:,:,j) = normc(featmap_t(:,:,j));
        end
        featmap_t = reshape(featmap_t, [], 16);
        maxpool_tr_preL2(:,i) = max(featmap_t, [], 2);
    else
        featmap_t = reshape(featmap_t, [], 16);
        maxpool_tr(:,i) = max(featmap_t, [], 2);
    end
end

parfor i = 1 : size(avepool_te, 2)
    featmap_t = reshape(vggpool5_te(:,:,:,:,i), [7*10, 512, 16]);
    featmap_t = permute(featmap_t, [2 1 3]);
    if preL2
        for j = 1 : 16
            featmap_t(:,:,j) = normc(featmap_t(:,:,j));
        end
        featmap_t = reshape(featmap_t, [], 16);
        maxpool_te_preL2(:,i) = max(featmap_t, [], 2);
    else
        featmap_t = reshape(featmap_t, [], 16);
        maxpool_te(:,i) = max(featmap_t, [], 2);
    end
end

% model = mpsvm.train(label_train, sparse(double(maxpool_tr_norm)), '-s 0 -n 32 -c 10 -q', 'col');
% [predicted_label] = mpsvm.predict(label_test, sparse(double(maxpool_te_norm)), model,'', 'col');

model_vgg19_max_preL2 = mpsvm.train(label_train, sparse(double(maxpool_tr_preL2)), '-s 0 -n 32 -c 10 -q', 'col');
[predicted_label] = mpsvm.predict(label_test, sparse(double(maxpool_te_preL2)), model,'', 'col');