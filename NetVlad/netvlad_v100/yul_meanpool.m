parfor i = 1 : size(avepool_tr, 2)
    featmap_t = reshape(vggpool5_tr(:,:,:,:,i), [7*10, 512, 16]);
    featmap_t = reshape(permute(featmap_t, [2 1 3]), [], 16);
    avepool_tr(:,i) = mean(featmap_t, 2);
end

parfor i = 1 : size(avepool_te, 2)
    featmap_t = reshape(vggpool5_te(:,:,:,:,i), [7*10, 512, 16]);
    featmap_t = reshape(permute(featmap_t, [2 1 3]), [], 16);
    avepool_te(:,i) = mean(featmap_t, 2);
end