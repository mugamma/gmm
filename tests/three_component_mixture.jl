@testset "correctly infers three component mixture" begin
    ys = [11.26, 28.93, 30.52, 30.09, 29.46, 10.03, 11.24, 11.55, 30.4, -18.44,
          10.91, 11.89, -20.64, 30.59, 14.84, 13.54, 7.25, 12.83, 11.86, 29.95,
          29.47, -18.16, -19.28, -18.87, 9.95, 28.24, 9.43, 7.38, 29.46, 30.73,
          7.75, 28.29, -21.99, -20.0, -20.86, 15.5, -18.62, 13.11, 28.66,
          28.18, -18.78, -20.48, 9.18, -20.12, 10.2, 30.26, -14.94, 5.45, 31.1,
          30.01, 10.52, 30.48, -20.37, -19.3, -21.92, -18.31, -18.9, -20.03,
          29.32, -17.53, 10.61, 6.38, -20.72, 10.29, 11.21, -18.98, 8.57,
          10.47, -22.4, 6.58, 29.8, -17.43, 7.8, 9.72, -21.53, 11.76, 29.72,
          29.31, 6.82, 15.51, 10.69, 29.56, 8.84, 30.93, 28.75, 10.72, 9.21,
          8.57, 11.92, -23.96, -19.78, -17.2, 11.79, 29.95, 7.29, 6.57, -17.99,
          13.29, -22.53, -20.0]
    zs = [2, 3, 3, 3, 3, 2, 2, 2, 3, 1, 2, 2, 1, 3, 2, 2, 2, 2, 2, 3, 3, 1, 1,
          1, 2, 3, 2, 2, 3, 3, 2, 3, 1, 1, 1, 2, 1, 2, 3, 3, 1, 1, 2, 1, 2, 3,
          1, 2, 3, 3, 2, 3, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1,
          2, 3, 1, 2, 2, 1, 2, 3, 3, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 2, 1, 1, 1,
          2, 3, 2, 2, 1, 2, 1, 1]
    @assert length(ys) == length(zs)
    n = length(zs)
    μs = [-20.0, 10.0, 30.0]
    σ²s = [3.0, 5.0, 1.0]
    tr, = generate(gmm, (n,), make_constraints(ys, μs, σ²s, zs))

    mcmc_tr = initial_trace(ys, 3, 2)
    best_tr = mcmc_tr
    for i=1:100
        print('.')
        mcmc_tr = mcmc_kernel(mcmc_tr)
        best_tr = get_score(mcmc_tr) > get_score(best_tr) ? mcmc_tr : best_tr
    end
    println()

    ρ = sort(1:3, by=i->best_tr[:μ => i]) # best_tr permutation
    permuted_zs = ρ[map(i->best_tr[:z => i], 1:n)]
    permuted_μs = map(i->best_tr[:μ => ρ[i]], 1:best_tr[:k])
    permuted_σ²s = map(i->best_tr[:σ² => ρ[i]], 1:best_tr[:k])
    permuted_ch = make_constraints(ys, permuted_μs, permuted_σ²s, permuted_zs)
    permuted_tr, = generate(gmm, (n,), permuted_ch)

    @test abs(1 - get_score(permuted_tr) / get_score(tr)) < 0.1

    @test permuted_tr[:k] == 3
    @test abs(permuted_tr[:μ => 1]- -20.0) < 1
    @test abs(permuted_tr[:μ => 2]- 10.0) < 1
    @test abs(permuted_tr[:μ => 3]- 30.0) < 1
    @test abs(permuted_tr[:σ² => 1] - 3.0) < 2
    @test abs(permuted_tr[:σ² => 2] - 5.0) < 2
    @test abs(permuted_tr[:σ² => 3] - 1.0) < 2

    n_misallocated = sum([zs[i] != permuted_tr[:z => i] for i=1:n])
    println("$n_misallocated samples misallocated")
    @test n_misallocated < 0.05 * n
end
