module MCBot

include("board.jl")

export Color, Empty, Black, White
export Point, P, other, neighbors, with_neighbors
export Board, liberties, on_board, off_board, points, valid_moves, play
export print_board, print_board_history
export BoardTree, deepen_tree, best_move, self_play

using .BoardModule

using Test: @test

mutable struct BoardTree
    board::Board
    player::Color
    value::Int16
    move::Dict{Point, BoardTree}
end

function BoardTree(board::Board, player::Color)
    BoardTree(board, player, eval_board(board, player), Dict{Point, BoardTree}())
end

function eval_board(board::Board, player::Color)
    val = sum(points(board)) do p
        if board[p] == player
            return liberties(board, p)
        elseif board[p] == other(player)
            return -liberties(board, p)
        else
            return 0
        end
    end
    Int16(val)
end

function test_eval_board()
    b = Board(9)
    @test eval_board(b, Black) == 0
    @test eval_board(b, White) == 0
    b[P(1, 1)] = Black
    @test eval_board(b, Black) == 2
    @test eval_board(b, White) == -2
    b[P(1, 2)] = White
    @test eval_board(b, Black) == -1
    @test eval_board(b, White) == 1
    b[P(1, 3)] = Black
    @test eval_board(b, Black) == 2
    @test eval_board(b, White) == -2
    b[P(1, 4)] = Black
    @test eval_board(b, Black) == 6
    @test eval_board(b, White) == -6
end

function Base.length(tree::BoardTree)
    if length(tree.move) == 0
        return 1
    end
    1 + sum(length(subtree) for subtree in values(tree.move))
end

function deepen_tree(tree::BoardTree)
    vms = valid_moves(tree.board, tree.player)
    if length(vms) == 0
        return tree.value
    end
    random_move = rand(vms)
    if haskey(tree.move, random_move)
        deepen_tree(tree.move[random_move])
    else
        next_board = deepcopy(tree.board)
        play(next_board, random_move, tree.player)
        next_board_tree = BoardTree(next_board, other(tree.player))
        tree.move[random_move] = next_board_tree
    end
    tree.value = minimum(-t.value for t in values(tree.move))
end

function best_move(tree::BoardTree)
    best_value = minimum(t.value for t in values(tree.move))
    best_moves = filter(p -> tree.move[p].value == best_value, keys(tree.move))
    rand(best_moves)
end

function test_board_tree()
    b = Board(2)
    t = BoardTree(b, Black)
    @test length(t) == 1
    deepen_tree(t)
    @test length(t) == 2
    deepen_tree(t)
    @test length(t) == 3
    for _ in 1:100
        deepen_tree(t)
    end
    @test length(t) > 3
    @test best_move(t) in valid_moves(b, Black)
end

function self_play()
    board = Board(9)
    current_player = Black
    board_tree = BoardTree(board, current_player)
    while length(valid_moves(board, current_player)) > 0
        start_time = time()
        while time() - start_time < 6
            deepen_tree(board_tree)
        end
        board_tree = board_tree.move[best_move(board_tree)]
        board = board_tree.board
        current_player = other(current_player)
        @assert board_tree.player == current_player
        print_board(board)
        println(length(board_tree))
    end
end

function test_all()
    test_eval_board()
    test_board_tree()
end

test_all()

end
