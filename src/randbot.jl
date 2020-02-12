module RandomPlayerModule

export RandomPlayer, move

include("board.jl")

using .BoardModule

struct RandomPlayer; end

move(player::RandomPlayer, board, current_player) = rand(valid_moves(board, current_player))

end # module
