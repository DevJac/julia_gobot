include("board.jl")

using .BoardModule

using Test: @test
using Flux
using BSON: @save, @load

@test begin
    b = Board(9)
    b[P(1, 1)] = Black
    b[P(1, 1)] == Black
end

function encode_board(board::Board, color::Color)::Array{Int8, 4}
    valid_move_set = Set(valid_moves(board, color))
    t = zeros(Int8, board.size, board.size, 11, 1)
    for p in points(board)
        t[p.x, p.y, 1] = board[p] == Black && liberties(board, p) == 1
        t[p.x, p.y, 2] = board[p] == Black && liberties(board, p) == 2
        t[p.x, p.y, 3] = board[p] == Black && liberties(board, p) == 3
        t[p.x, p.y, 4] = board[p] == Black && liberties(board, p) > 3
        t[p.x, p.y, 5] = board[p] == White && liberties(board, p) == 1
        t[p.x, p.y, 6] = board[p] == White && liberties(board, p) == 2
        t[p.x, p.y, 7] = board[p] == White && liberties(board, p) == 3
        t[p.x, p.y, 8] = board[p] == White && liberties(board, p) > 3
        t[p.x, p.y, 9] = color == Black
        t[p.x, p.y, 10] = color == White
        t[p.x, p.y, 11] = p in valid_move_set
    end
    return t
end

@test begin
    b = Board(9)
    play(b, P(1, 1), Black)
    @assert encode_board(b, Black)[1, 1, 2] == 1
    encode_board(b, Black)[1, 1, 3] == 0
end

function create_model(board::Board)
    function (x)
        # Individual layers
        conv1 = Conv((3, 3), 11=>50, relu)
        conv2 = Conv((3, 3), 50=>50, relu)
        conv3 = Conv((3, 3), 50=>50, relu)
        conv4 = Conv((3, 3), 50=>50, relu)
        processed_board = Dense(50, 500)
        policy_hidden_layer = Dense(500, 500, relu)
        policy_output_layer = Dense(500, Int16(board.size)^2)
        value_hidden_layer = Dense(500, 500, relu)
        value_output_layer = Dense(500, 1, sigmoid)

        # Assembled layers
        conv_chain = Chain(
            conv1,
            conv2,
            conv3,
            conv4,
            (a) -> reshape(a, (50, 1)),
            processed_board)
        policy_chain = Chain(
            policy_hidden_layer,
            policy_output_layer,
            softmax,
            (a) -> reshape(a, (board.size, board.size)))
        value_chain = Chain(
            value_hidden_layer,
            value_output_layer)
        conv_chain_output = conv_chain(x)
        return (policy_chain(conv_chain_output), value_chain(conv_chain_output)[1])
    end
end

@test begin
    b = Board(9)
    m = create_model(b)
    y_policy, y_value = m(encode_board(b, Black))
    @assert size(y_policy) == (b.size, b.size)
    size(y_value) == ()
end

struct MoveMemory
    board::Board
    color::Color
    move::Point
end

mutable struct NNBot
    model
    move_memory::Array{MoveMemory}
end

function NNBot(board::Board)
    NNBot(create_model(board), Array{MoveMemory}[])
end

function genmove_random(bot::NNBot, board::Board, color::Color)
    valid_move_set = Set(valid_moves(board, color))
    if length(valid_move_set) == 0
        return true, nothing
    end
    random_move = rand(valid_move_set)
    push!(bot.move_memory, MoveMemory(deepcopy(board), color, random_move))
    return false, random_move
end

function genmove_intuition(bot::NNBot, board::Board, color::Color)
    valid_move_set = Set(valid_moves(board, color))
    if length(valid_move_set) == 0
        return true, nothing
    end
    policy, value = bot.model(encode_board(board, color))
    best_move = nothing
    best_move_policy_value = -99.9f0
    for i in eachindex(policy)
        move = P(i[1], i[2])
        policy_value = policy[i]
        if move in valid_move_set && policy_value > best_move_policy_value
            best_move_policy_value = policy_value
            best_move = move
        end
    end
    return false, best_move
end

function report_winner(bot::NNBot, winning_color::Color)
    X = Array{Int8, 4}[]
    Y_policy = Array{Int8, 2}[]
    Y_value = Int8[]
    for move in bot.move_memory
        won = move.color == winning_color
        correct_policy = zeros(Int8, move.board.size, move.board.size)
        correct_policy[move.move] = won ? 1 : -1
        push!(X, encode_board(move.board, move.color))
        push!(Y_policy, correct_policy)
        push!(Y_value, won ? Int8(1) : Int8(0))
    end
    X = cat(X, dims=5)
    Y_policy = cat(Y_policy, dims=2)
    Y_value = cat(Y_value, dims=1)
    bot.move_memory = Array{MoveMemory}[]
    game_id = join(rand("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in 1:6)
    game_file = "games/" * game_id * ".game"
    @save game_file X Y_policy Y_value
end

function self_play_single_game()
    board = Board(9)
    bot = NNBot(board)
    print_board(board)
    while true
        # Black's move
        resign, move = genmove_intuition(bot, board, Black)
        if resign
            println("White Wins!")
            report_winner(bot, White)
            break
        end
        play(board, move, Black)
        print_board(board)
        resign, move = genmove_intuition(bot, board, White)
        if resign
            println("Black Wins!")
            report_winner(bot, Black)
            break
        end
        play(board, move, White)
        print_board(board)
    end
end

for _ in 1:1
    self_play_single_game()
end
