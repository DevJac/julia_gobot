using Distributed

@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using GameRunner
@everywhere using JLD
@everywhere using MCPlayerM
@everywhere using Printf
@everywhere using ProgressMeter

@showprogress pmap((i, ARGS[2]) for i in 1:parse(Int, ARGS[1])) do (i, save_to)
    game_record = play_game(
        MCPlayer(),
        MCPlayer(),
        board_size=7,
        quiet=true)
    save(@sprintf("%s-%04d.jld", save_to, i), "game_record", game_record)
end
