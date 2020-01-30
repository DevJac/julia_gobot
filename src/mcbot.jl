module MCBot

include("board.jl")

export Color, Empty, Black, White
export Point, P, other, neighbors, with_neighbors
export Board, liberties, on_board, off_board, points, valid_moves, play
export print_board, print_board_history
export BoardTree, rollout, best_move, self_play

using .BoardModule

using Printf
using Random
using Test: @test

mutable struct BoardTree
    board::Board
    player::Color
    black_wins::Int32
    white_wins::Int32
    move::Dict{Point, BoardTree}
end

function BoardTree(board::Board, player::Color)
    BoardTree(board, player, 0, 0, Dict{Point, BoardTree}())
end

function Base.length(tree::BoardTree)
    if length(tree.move) == 0
        return 1
    end
    1 + sum(length(subtree) for subtree in values(tree.move))
end

function rollout_with_random_moves(tree::BoardTree, valid_moves)
    rand(valid_moves)
end

function rollout_with_uct_moves(tree::BoardTree, valid_moves)
    c = 1.5
    shuffle!(valid_moves)
    N = if length(tree.move) == 0
        0
    else
        sum(values(tree.move)) do t
            t.black_wins + t.white_wins
        end
    end
    valid_moves_ucs = map(valid_moves) do p
        if !haskey(tree.move, p)
            Inf
        else
            t = tree.move[p]
            n = t.black_wins + t.white_wins
            w = tree.player == Black ? t.black_wins / n : t.white_wins / n
            w + c * sqrt(log(N) / n)
        end
    end
    valid_moves[argmax(valid_moves_ucs)]
end

function rollout(tree::BoardTree, move_selection=rollout_with_uct_moves)
    vms = valid_moves(tree.board, tree.player)
    if length(vms) == 0
        winner = other(tree.player)
        if winner == Black
            tree.black_wins += 1
        else
            tree.white_wins += 1
        end
        return winner
    end
    selected_move = move_selection(tree, vms)
    if !haskey(tree.move, selected_move)
        next_board = deepcopy(tree.board)
        play(next_board, selected_move, tree.player)
        tree.move[selected_move] = BoardTree(next_board, other(tree.player))
    end
    rollout_winner = rollout(tree.move[selected_move], rollout_with_random_moves)
    if rollout_winner == Black
        tree.black_wins += 1
    else
        tree.white_wins += 1
    end
    rollout_winner
end

function best_move(tree::BoardTree)
    moves = collect(tree.move)
    visit_count = [t.black_wins + t.white_wins for (_, t) in moves]
    r = moves[argmax(visit_count)]
    @info "Best move rollouts" r.second.black_wins r.second.white_wins
    r.first
end

function test_board_tree()
    b = Board(2)
    t = BoardTree(b, Black)
    @test length(t) == 1
    rollout(t)
    @test length(t) > 1
    @test best_move(t) in valid_moves(b, Black)
end

function self_play()
    board = Board(5)
    current_player = Black
    board_tree = BoardTree(board, current_player)
    while length(valid_moves(board, current_player)) > 0
        start_time = time()
        while time() - start_time < 6
            rollout(board_tree)
        end
        @printf("%s's move\n", current_player)
        board_tree = board_tree.move[best_move(board_tree)]
        board = board_tree.board
        current_player = other(current_player)
        @assert board_tree.player == current_player
        print_board(board)
        @printf("Tree size: %d\n\n", length(board_tree))
    end
end

function test_all()
    test_board_tree()
end

test_all()

end
