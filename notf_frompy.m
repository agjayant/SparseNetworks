function U = notf_frompy(R, L, k)

    addpath('~/SparseNetworks/tensor-factorization/bin/')
    
    temp1 = randn(k,k);
    lambda = randn(k,1);

    T = ktensor(lambda, {temp1, temp1, temp1});
    T = full(T);
    ind = 1;
    for i=1:k
        for j=1:k
            for p=1:k
                T(p,j,i) = R(ind);
                ind = ind +1;
            end
        end
    end

    U = no_tenfact(T, L, 3);
end
