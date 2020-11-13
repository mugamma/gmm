@testset "dirichlet generative function" begin
    dirichlet_logpdf(α, y) = Distributions.logpdf(Distributions.Dirichlet(α), y)

    @testset "has the right support" begin
        for _=1:5
            x = dirichlet(ones(10))
            @test all((x .< 1.0) .& (x .> 0))
            @test abs(sum(x) - 1.0) < 1e-15
        end
    end

    @testset "gives the right score" begin
        for _=1:5
            α = beta(1, 1) .* ones(10)
            tr, score = generate(dirichlet, (α,))
            x = get_retval(tr)
            #true_score = Distributions.logpdf(Distributions.Dirichlet(α), x)
            true_score = 0.0
            @test abs(score - true_score) < 1e-15
        end
    end

    @testset "constrains correctly" begin
        for _=1:5
            constrained_addrs = [i in (4, 20, 19, 9, 7) for i=1:20]
            α = ones(20)
            x = rand(Distributions.Dirichlet(α))
            constraints = choicemap([(i => x[i])
                                     for i=1:20 if constrained_addrs[i]]...)
            tr, score = generate(dirichlet, (α,), constraints)
            y = get_retval(tr)
            true_full = dirichlet_logpdf(α, y)
            true_unc = dirichlet_logpdf(α[.!constrained_addrs],
                                        y[.!constrained_addrs] ./
                                        sum(y[.!constrained_addrs]))
            true_score = true_full - true_unc
            @test all(x[constrained_addrs] == y[constrained_addrs])
            @test abs(score - true_score) < 1e-3
        end
    end

    @testset "updates correctly" begin
        α, constrained_addrs = ones(20), [10, 20, 3, 15]
        unconstrained_addrs = [i for i=1:20 if !(i in constrained_addrs)]
        tr1, score1 = generate(dirichlet, (α,))
        tr2, score2 = generate(dirichlet, (α,))
        x, y = get_retval(tr1), get_retval(tr2)
        correction = sum(x[constrained_addrs]) / sum(y[constrained_addrs])
        constraints = choicemap([(i, y[i] * correction)
                                 for i in constrained_addrs]...)
        tr3, w, retdiff, discard = update(tr1, (α,), (NoChange(),), constraints)
        z = get_retval(tr3)

        @test z[constrained_addrs] == y[constrained_addrs] * correction
        @test z[unconstrained_addrs] == x[unconstrained_addrs]
        @test abs(w - dirichlet_logpdf(α, z) + dirichlet_logpdf(α, x)) < 1e-12
        @test retdiff isa VectorDiff
        @test retdiff.new_length == retdiff.prev_length == 20
        @test all([a in keys(retdiff.updated) for a in constrained_addrs])
        @test all([has_value(discard, a) for a in constrained_addrs])
    end
end
