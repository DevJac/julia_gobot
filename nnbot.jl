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

function uid()
    join(rand("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in 1:6)
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
            (a) -> reshape(a, (50, size(x, 4))),
            processed_board)
        policy_chain = Chain(
            policy_hidden_layer,
            policy_output_layer,
            softmax,
            (a) -> reshape(a, (board.size, board.size, size(x, 4))))
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
    encoded_board = encode_board(b, Black)
    @assert size(encoded_board) == (b.size, b.size, 11, 1)
    y_policy, y_value = m(encoded_board)
    @assert size(y_policy) == (b.size, b.size, 1)
    size(y_value) == ()
end

struct MoveMemory
    board::Board
    color::Color
    move::Point
end

struct GameMemory
    move_memory::Array{MoveMemory}
    winning_color::Color
end

mutable struct NNBot
    model
    move_memory::Array{MoveMemory}
end

function NNBot(board::Board)
    if isfile("model.bson")
        @load "model.bson" model
        NNBot(model, MoveMemory[])
    else
        NNBot(create_model(board), MoveMemory[])
    end
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
    push!(bot.move_memory, MoveMemory(deepcopy(board), color, best_move))
    return false, best_move
end

function report_winner(bot::NNBot, winning_color::Color)
    game_memory = GameMemory(bot.move_memory, winning_color)
    bot.move_memory = MoveMemory[]
    game_id = uid()
    game_file = "games/" * game_id * ".game"
    @save game_file game_memory
    return game_memory
end

function train(bot::NNBot, game_memory::GameMemory)
    println("Training Prep")
    X = Array{Int8, 4}[]
    Y_policy = Array{Float32, 2}[]
    Y_value = Int8[]
    move_memory_length = length(game_memory.move_memory)
    for i in 1:move_memory_length
        move = game_memory.move_memory[i]
        won = move.color == game_memory.winning_color
        if i < move_memory_length
            _, v = bot.model(encode_board(game_memory.move_memory[i+1].board, move.color))
            local next_move_value = v.data
        else
            local next_move_value = won ? 1f0 : 0f0
        end
        policy, value = bot.model(encode_board(move.board, move.color))
        correct_policy = zeros(Float32, move.board.size, move.board.size)
        correct_policy[move.move] = (won ? 1f0 : -1f0) * abs(value.data - next_move_value)
        push!(X, encode_board(move.board, move.color))
        push!(Y_policy, correct_policy)
        push!(Y_value, won ? Int8(1) : Int8(0))
    end
    X = cat(X..., dims=4)
    Y_policy = cat(Y_policy..., dims=3)
    function loss(x, y)
        y_policy, y_value = bot.model(x)
        new_shape = (size(y[1])[1] * size(y[1])[2], size(y[1])[3])
        policy_loss = Flux.crossentropy(reshape(y_policy, new_shape), reshape(y[1], new_shape))
        value_loss = Flux.mse(y_value, y[2])
        return policy_loss + value_loss
    end
    println("Training.       Pre-training loss: ", loss(X, (Y_policy, Y_value)).data)
    Flux.train!(loss, params(bot.model), [(X, (Y_policy, Y_value))], Descent())
    println("Training Done. Post-training loss: ", loss(X, (Y_policy, Y_value)).data)
    sleep(1.5)
end

function save_model(bot::NNBot)
    model = bot.model
    temp_name = "model.bson." * uid()
    @save temp_name model
    mv(temp_name, "model.bson", force=true)
end

function self_play(n)
    for game in 1:n
        board = Board(9)
        bot = NNBot(board)
        print_board(board)
        game_memory = nothing
        while true
            # Black's move
            resign, move = genmove_intuition(bot, board, Black)
            if resign
                println("White Wins Game ", game)
                game_memory = report_winner(bot, White)
                break
            end
            play(board, move, Black)
            print_board(board)
            resign, move = genmove_intuition(bot, board, White)
            if resign
                println("Black Wins Game ", game)
                game_memory = report_winner(bot, Black)
                break
            end
            play(board, move, White)
            print_board(board)
        end
        train(bot, game_memory)
        save_model(bot)
    end
end

self_play(1)
self_play(99)
