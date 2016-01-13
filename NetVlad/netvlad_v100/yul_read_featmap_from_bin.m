function blob = yul_read_featmap_from_bin( list, mapsize )
    blob = zeros([mapsize, length(list)], 'single');
%     tic
    for i = 1 : length(list)
        blob(:,:,:,i) = readbin(list{i}, mapsize, 'single');
%         if mod(i, round(length(list)/100)) == 1
%             fprintf('Dealing with %d/%d...', i, length(list));
%             toc
%         end
    end
end

