parfor i = 1 : size(avepool_tr, 2)
    featmap_t = reshape(vggpool5_tr(:,:,:,:,i), [7*10, 512, 16]);
    featmap_t = reshape(permute(featmap_t, [2 1 3]), [], 16);
    maxpool_tr(:,i) = max(featmap_t');
end

parfor i = 1 : size(avepool_te, 2)
    featmap_t = reshape(vggpool5_te(:,:,:,:,i), [7*10, 512, 16]);
    featmap_t = reshape(permute(featmap_t, [2 1 3]), [], 16);
    maxpool_te(:,i) = max(featmap_t');
end

model = mpsvm.train(label_train, sparse(double(maxpool_tr)), '-s 0 -n 32 -c 10 -q', 'col');