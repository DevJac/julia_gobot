module Game

export play_game

include("board.jl")
include("randbot.jl")

using .BoardModule
using .RandomPlayerModule

using Printf
using Random

struct GameState
    board :: Board
    player :: Color
    move :: Point
end

struct GameRecord
    states :: Vector{GameState}
    winner :: Color
end

function play_game(p1, p2; board_size=19)
    players = Dict{Color, Any}()
    players[Black], players[White] = shuffle([p1, p2])
    game_states = GameState[]
    board = Board(board_size)
    current_player = Black
    while length(valid_moves(board, current_player)) > 0
        m = move(players[current_player], board, current_player)
        push!(game_states, GameState(deepcopy(board), current_player, m))
        play(board, m, current_player)
        @printf("%s's (%s) move: %s\n", current_player, players[current_player], m)
        print_board(board)
        current_player = other(current_player)
    end
    winner = other(current_player)
    @printf("%s wins\n", winner)
    GameRecord(game_states, winner)
end

function test_all()
    play_game(RandomPlayer(), RandomPlayer())
end

end # module
