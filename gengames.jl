using GameRunner
using JLD
using MCPlayerM
using Printf
using ProgressMeter
using Serialization

@showprogress 1 "Playing games..." for i in 1:parse(Int, ARGS[1])
    game_record = play_game(
        MCPlayer(),
        GameRunner.RandomPlayerM.RandomPlayer(),
        board_size=9,
        quiet=true)
    save(@sprintf("%s-%06d.jld", ARGS[2], i), "game_record", game_record)
end
