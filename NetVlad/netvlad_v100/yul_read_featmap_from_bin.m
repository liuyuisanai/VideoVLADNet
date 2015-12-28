function blob = yul_read_featmap_from_bin( list, mapsize )
    blob = zeros([mapsize, length(list)], 'single');
    for i = 1 : length(list)
        blob(:,:,:,i) = readbin(list{i}, mapsize, 'single');
    end
end

