@testset "index shift calculations" begin
    @testset "split index shifts 1" begin
        k, j, j1, j2 = 7, 1, 6, 4
        # before a1 a2 a3 a4 a5 a6 a7
        # after  a2 a3 a4 x1 a5 x2 a6 a7
        @test_throws ArgumentError split_idx(1, k, j, j1, j2)
        @test split_idx(2, k, j, j1, j2) == 1
        @test split_idx(3, k, j, j1, j2) == 2
        @test split_idx(4, k, j, j1, j2) == 3
        @test split_idx(5, k, j, j1, j2) == 5
        @test split_idx(6, k, j, j1, j2) == 7
        @test split_idx(7, k, j, j1, j2) == 8
    end

    @testset "split index shifts 2" begin
        k, j, j1, j2 = 7, 4, 3, 4
        # before a1 a2 a3 a4 a5 a6 a7
        # before a1 a2 x1 x2 a3 a5 a6 a7
        @test split_idx(1, k, j, j1, j2) == 1
        @test split_idx(2, k, j, j1, j2) == 2
        @test split_idx(3, k, j, j1, j2) == 5
        @test_throws ArgumentError split_idx(4, k, j, j1, j2)
        @test split_idx(5, k, j, j1, j2) == 6
        @test split_idx(6, k, j, j1, j2) == 7
        @test split_idx(7, k, j, j1, j2) == 8
    end

    @testset "split index shifts 3" begin
        k, j, j1, j2 = 7, 4, 6, 3
        # before a1 a2 a3 a4 a5 a6 a7
        # after  a1 a2 x2 a3 a5 x1 a6 a7
        @test split_idx(1, k, j, j1, j2) == 1
        @test split_idx(2, k, j, j1, j2) == 2
        @test split_idx(3, k, j, j1, j2) == 4
        @test_throws ArgumentError split_idx(4, k, j, j1, j2)
        @test split_idx(5, k, j, j1, j2) == 5
        @test split_idx(6, k, j, j1, j2) == 7
        @test split_idx(7, k, j, j1, j2) == 8
    end

    @testset "split index shifts 4" begin
        k, j, j1, j2 = 7, 7, 7, 8
        # before a1 a2 a3 a4 a5 a6 a7
        # after  a1 a2 a3 a4 a5 a6 x1 x2
        @test split_idx(1, k, j, j1, j2) == 1
        @test split_idx(2, k, j, j1, j2) == 2
        @test split_idx(3, k, j, j1, j2) == 3
        @test split_idx(4, k, j, j1, j2) == 4
        @test split_idx(5, k, j, j1, j2) == 5
        @test split_idx(6, k, j, j1, j2) == 6
        @test_throws ArgumentError split_idx(7, k, j, j1, j2)
    end

    @testset "merge index shifts 1" begin
        k, j, j1, j2 = 7, 3, 3, 5
        # before a1 a2 a3 a4 a5 a6 a7
        # after  a1 a2 xx a4 a6 a7
        @test merge_idx(1, k, j, j1, j2) == 1
        @test merge_idx(2, k, j, j1, j2) == 2
        @test_throws ArgumentError merge_idx(3, k, j, j1, j2)
        @test merge_idx(4, k, j, j1, j2) == 4
        @test_throws ArgumentError merge_idx(5, k, j, j1, j2)
        @test merge_idx(6, k, j, j1, j2) == 5
        @test merge_idx(7, k, j, j1, j2) == 6
    end

    @testset "merge index shifts 2" begin
        k, j, j1, j2 = 7, 6, 6, 7
        # before a1 a2 a3 a4 a5 a6 a7
        # after  a1 a2 a3 a4 a5 xx
        @test merge_idx(1, k, j, j1, j2) == 1
        @test merge_idx(2, k, j, j1, j2) == 2
        @test merge_idx(3, k, j, j1, j2) == 3
        @test merge_idx(4, k, j, j1, j2) == 4
        @test merge_idx(5, k, j, j1, j2) == 5
        @test_throws ArgumentError merge_idx(6, k, j, j1, j2)
        @test_throws ArgumentError merge_idx(7, k, j, j1, j2)
    end

    @testset "merge index shifts 3" begin
        k, j, j1, j2 = 7, 3, 4, 6
        # before a1 a2 a3 a4 a5 a6 a7
        # after  a1 a2 xx a3 a5 a7
        @test merge_idx(1, k, j, j1, j2) == 1
        @test merge_idx(2, k, j, j1, j2) == 2
        @test merge_idx(3, k, j, j1, j2) == 4
        @test_throws ArgumentError merge_idx(4, k, j, j1, j2)
        @test merge_idx(5, k, j, j1, j2) == 5
        @test_throws ArgumentError merge_idx(6, k, j, j1, j2)
        @test merge_idx(7, k, j, j1, j2) == 6
    end

    @testset "merge index shifts 4" begin
        k, j, j1, j2 = 7, 1, 1, 2
        # before a1 a2 a3 a4 a5 a6 a7
        # after  xx a3 a4 a5 a6 a7
        @test_throws ArgumentError merge_idx(1, k, j, j1, j2)
        @test_throws ArgumentError merge_idx(2, k, j, j1, j2)
        @test merge_idx(3, k, j, j1, j2) == 2
        @test merge_idx(4, k, j, j1, j2) == 3
        @test merge_idx(5, k, j, j1, j2) == 4
        @test merge_idx(6, k, j, j1, j2) == 5
        @test merge_idx(7, k, j, j1, j2) == 6
    end
end
