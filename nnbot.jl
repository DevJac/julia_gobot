include("board.jl")

using .BoardModule

using Test: @test
using Flux
using BSON: @save, @load
using ProgressMeter

b = Board(9)
b[P(1, 1)] = Black
@test b[P(1, 1)] == Black

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

b = Board(9)
play(b, P(1, 1), Black)
@test encode_board(b, Black)[1, 1, 2] == 1
@test encode_board(b, Black)[1, 1, 3] == 0

function create_model(board_size)
    board_size = Int64(board_size)
    # Individual layers
    upper_network_size = 40
    lower_network_size = 200
    conv1 = Conv((3, 3), 11=>upper_network_size, relu, pad=(1, 1))
    conv2 = Conv((3, 3), upper_network_size=>upper_network_size, relu, pad=(1, 1))
    conv3 = Conv((3, 3), upper_network_size=>upper_network_size, relu, pad=(1, 1))
    conv4 = Conv((3, 3), upper_network_size=>upper_network_size, relu, pad=(1, 1))
    processed_board = Dense(upper_network_size*board_size^2, lower_network_size)
    policy_hidden_layer = Dense(lower_network_size, lower_network_size, relu)
    policy_output_layer = Dense(lower_network_size, board_size^2)
    value_hidden_layer = Dense(lower_network_size, lower_network_size, relu)
    value_output_layer = Dense(lower_network_size, 1, sigmoid)

    # Assembled layers
    conv_chain = Chain(
        conv1,
        conv2,
        conv3,
        conv4,
        (a) -> reshape(a, upper_network_size*board_size^2),
        processed_board)
    policy_chain = Chain(
        policy_hidden_layer,
        policy_output_layer,
        softmax,
        (a) -> reshape(a, (board_size, board_size)))
    value_chain = Chain(
        value_hidden_layer,
        value_output_layer)

    function (x)
        conv_chain_output = conv_chain(x)
        policy_output = policy_chain(conv_chain_output)
        @assert !any(isnan(n) for n in policy_output)
        value_output = value_chain(conv_chain_output)[1]
        return (policy_output, value_output)
    end
end

b = Board(25)
m = create_model(b.size)
encoded_board = encode_board(b, Black)
@test size(encoded_board) == (b.size, b.size, 11, 1)
y_policy, y_value = m(encoded_board)
@test size(y_policy) == (b.size, b.size)
@test size(y_value) == ()
y_policy2, y_value2 = m(encoded_board)
@test y_policy == y_policy2
@test y_value == y_value2

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

function NNBot(board_size)
    if isfile("model.bson")
        @load "model.bson" model
        NNBot(model, MoveMemory[])
    else
        NNBot(create_model(board_size), MoveMemory[])
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
        policy_value = policy[i] + rand()
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

function train(model, game_memories::Array{GameMemory})
    X = Array{Int8, 4}[]
    Y_policy = Array{Float32, 2}[]
    Y_value = Int8[]
    @showprogress 1 "Training Prep " for game_memory in game_memories
        move_memory_length = length(game_memory.move_memory)
        for i in 1:move_memory_length
            move = game_memory.move_memory[i]
            won = move.color == game_memory.winning_color
            if i < move_memory_length
                _, v = model(encode_board(game_memory.move_memory[i+1].board, move.color))
                local next_move_value = v.data
            else
                local next_move_value = won ? 1f0 : 0f0
            end
            policy, value = model(encode_board(move.board, move.color))
            correct_policy = zeros(Float32, move.board.size, move.board.size)
            correct_policy[move.move] = (won ? 1f0 : -1f0) * abs(value.data - next_move_value)
            push!(X, encode_board(move.board, move.color))
            push!(Y_policy, correct_policy)
            push!(Y_value, won ? Int8(1) : Int8(0))
        end
    end
    data = [(X[i], (Y_policy[i], Y_value[i])) for i in 1:length(X)]
    function loss(x, y)
        y_policy, y_value = model(x)
        new_shape = size(y[1])[1] * size(y[1])[2]
        policy_loss = Flux.logitcrossentropy(reshape(y_policy, new_shape), reshape(y[1], new_shape))
        value_loss = Flux.mse(y_value, y[2])
        return policy_loss + value_loss
    end
    pre_training_loss = sum(loss(X, (Y_policy, Y_value)).data for (X, (Y_policy, Y_value)) in data) / length(data)
    println("Training.       Pre-training loss: ", pre_training_loss)
    opt = Descent()
    Flux.train!(loss, params(model.conv_chain, model.policy_chain, model.value_chain), data, opt)
    post_training_loss = sum(loss(X, (Y_policy, Y_value)).data for (X, (Y_policy, Y_value)) in data) / length(data)
    println("Training Done. Post-training loss: ", post_training_loss)
    return (pre_training_loss, post_training_loss)
end

function save_model(model)
    temp_name = "model.bson." * uid()
    @save temp_name model
    mv(temp_name, "model.bson", force=true)
end

function self_play(n)
    game_memories = GameMemory[]
    bot = NNBot(Int16(9))
    for game in 1:n
        board = Board(9)
        print_board(board)
        game_memory = nothing
        while true
            # Black's move
            resign, move = genmove_intuition(bot, board, Black)
            if resign
                println("White Wins Game ", game)
                push!(game_memories, report_winner(bot, White))
                break
            end
            play(board, move, Black)
            print_board(board)
            resign, move = genmove_intuition(bot, board, White)
            if resign
                println("Black Wins Game ", game)
                push!(game_memories, report_winner(bot, Black))
                break
            end
            play(board, move, White)
            print_board(board)
        end
    end
    pre_loss, post_loss = train(bot.model, game_memories)
    save_model(bot.model)
    return pre_loss, post_loss
end

while true
    losses = self_play(20)
    open("loss.txt", "a") do file
        println(file, losses)
    end
end
