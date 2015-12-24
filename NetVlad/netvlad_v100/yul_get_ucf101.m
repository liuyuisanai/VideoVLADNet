function db = yul_get_ucf101(paths, listf)
    db.name = sprintf('ucf101_%s', listf(1:end-4));
    annodir = fullfile(paths.dsetSpecDir, 'ucf101', listf);
    list = textread(annodir, '%s');
    if ~isempty(strfind(listf, 'train'))
        db.label = list(2:2:end);
        db.list = list(1:2:end);
        assert(numel(db.label) ~= numel(db.list), 'DB load error: list and label size not match!');
    else
        db.list = list;
    end
end