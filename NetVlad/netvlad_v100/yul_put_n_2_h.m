function concat_map = yul_put_n_2_h(map4d)
    concat_map = reshape(permute(map4d, [1 4 2 3 5]), [size(map4d, 1)*size(map4d, 4), size(map4d, 2), size(map4d, 3), size(map4d, 5)]);
end