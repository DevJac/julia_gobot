using BoardM
using GameRunner
using JLD

replay_game(load(ARGS[1], "game_record"))
