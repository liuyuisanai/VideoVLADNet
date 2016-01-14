preL2 = false;

if preL2
    meanpool_tr_preL2 = zeros(7*10*512, 9537, 'single');
    meanpool_te_preL2 = zeros(7*10*512, 3779, 'single');
else
    meanpool_tr = zeros(7*10*512, 9537, 'single');
    meanpool_te = zeros(7*10*512, 3779, 'single');
end


parfor i = 1 : size(avepool_tr, 2)
    featmap_t = reshape(vggpool5_tr(:,:,:,:,i), [7*10, 512, 16]);
    featmap_t = permute(featmap_t, [2 1 3]);
    if preL2
        for j = 1 : 16
            featmap_t(:,:,j) = normc(featmap_t(:,:,j));
        end
        featmap_t = reshape(featmap_t, [], 16);
        meanpool_tr_preL2(:,i) = max(featmap_t, [], 2);
    else
        featmap_t = reshape(featmap_t, [], 16);
        meanpool_tr(:,i) = mean(featmap_t, 2);
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
        meanpool_te_preL2(:,i) = max(featmap_t, [], 2);
    else
        featmap_t = reshape(featmap_t, [], 16);
        meanpool_te(:,i) = mean(featmap_t, 2);
    end
end

model = mpsvm.train(label_train, sparse(double(meanpool_tr)), '-s 0 -n 32 -c 10 -q', 'col');
fprintf('Pure Mean Pooling Result: ')
[predicted_label] = mpsvm.predict(label_test, sparse(double(meanpool_te)), model,'', 'col');

meanpool_te_norm = normc(meanpool_te);
meanpool_tr_norm = normc(meanpool_tr);
model = mpsvm.train(label_train, sparse(double(meanpool_tr_norm)), '-s 0 -n 32 -c 10 -q', 'col');
fprintf('Mean Pooling with Norm Result: ')
[predicted_label] = mpsvm.predict(label_test, sparse(double(meanpool_te_norm)), model,'', 'col');