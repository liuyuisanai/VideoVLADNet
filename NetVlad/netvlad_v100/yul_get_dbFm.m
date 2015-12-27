function db = yul_get_dbFm(dbSet, paths)
    db.numVideos = dbSet.numVideos;
    db.path = arrayfun(@(i_t)fullfile(paths.FmRootUCF101, ['vgg16_conv5_3_ucf101_train01_', num2str(i_t) '.bin']), ...
        1:db.numVideos, 'UniformOutput', false);
    if isfield(dbSet, 'label')
        db.label = cellfun(@(lab)str2num(lab), dbSet.label);
    end
end