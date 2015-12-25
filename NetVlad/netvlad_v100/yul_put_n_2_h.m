function concat_map = yul_put_n_2_h(map4d)
    framesNum = size(map4d, 4);
    if isa(map4d, 'gpuArray')
        concat_map = gpuArray(zeros(size(map4d, 1)*size(map4d, 4),size(map4d, 2),size(map4d, 3), 'single'));
    else
        concat_map = zeros(size(map4d, 1)*framesNum,size(map4d, 2),size(map4d, 3), 'single');
    end
    for i = 1 : framesNum
        concat_map((i-1)*size(map4d, 1)+1:i*size(map4d, 1),:,:,1) = map4d(:,:,:,i);
    end
end