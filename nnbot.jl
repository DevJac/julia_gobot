include("board.jl")

using .BoardModule

using Test: @test
using Flux
using BSON: @save, @load
using ProgressMeter
using Random
using Printf

b = Board(9)
b[P(1, 1)] = Black
@test b[P(1, 1)] == Black

function uid()
    join(rand("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in 1:6)
end

const encoded_board_channels = 9

function encode_board(board::Board, color::Color)::Array{Int8, 4}
    valid_move_set = Set(valid_moves(board, color))
    t = zeros(Int8, board.size, board.size, encoded_board_channels, 1)
    for p in points(board)
        t[p.x, p.y, 1] = board[p] == color
        t[p.x, p.y, 2] = board[p] == color && liberties(board, p) == 1
        t[p.x, p.y, 3] = board[p] == color && liberties(board, p) == 2
        t[p.x, p.y, 4] = board[p] == color && liberties(board, p) > 2
        t[p.x, p.y, 5] = board[p] == other(color)
        t[p.x, p.y, 6] = board[p] == other(color) && liberties(board, p) == 1
        t[p.x, p.y, 7] = board[p] == other(color) && liberties(board, p) == 2
        t[p.x, p.y, 8] = board[p] == other(color) && liberties(board, p) > 2
        t[p.x, p.y, 9] = p in valid_move_set
    end
    return t
end

b = Board(9)
play(b, P(1, 1), Black)
@test encode_board(b, Black)[1, 1, 1] == 1
@test encode_board(b, Black)[1, 1, 2] == 0
@test encode_board(b, Black)[1, 1, 3] == 1

function create_model(board_size)
    board_size = Int64(board_size)
    # Individual layers
    upper_network_size = 50
    lower_network_size = 500
    conv1 = Conv((3, 3), encoded_board_channels=>upper_network_size, relu, pad=(1, 1))
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

b = Board(3)
m = create_model(b.size)
encoded_board = encode_board(b, Black)
@test size(encoded_board) == (b.size, b.size, encoded_board_channels, 1)
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

function game_memories_to_data(model, game_memories::Array{GameMemory})
    data = Tuple{Array{Int8,4},Tuple{Array{Float32,2},Int8}}[]
    @showprogress 1 "Training Prep " for game_memory in game_memories
        move_memory_length = length(game_memory.move_memory)
        for i in 1:move_memory_length
            move = game_memory.move_memory[i]
            won = move.color == game_memory.winning_color
            if i <= move_memory_length-2
                next_move = game_memory.move_memory[i+2]
                @assert move.color == next_move.color
                # TODO: We encode boards twice or more. We should cache the encoded boards.
                _, v = model(encode_board(next_move.board, move.color))
                next_move_value = v.data
            else
                next_move_value = won ? 1f0 : 0f0
            end
            encoded_board = encode_board(move.board, move.color)
            policy, value = model(encoded_board)
            correct_policy = zeros(Float32, move.board.size, move.board.size)
            correct_policy[move.move] = (won ? 1f0 : -1f0) * abs(value.data - next_move_value)
            x = encoded_board
            y_policy = correct_policy
            y_value = won ? Int8(1) : Int8(0)
            push!(data, (x, (y_policy, y_value)))
        end
    end
    return data
end

function train(model, opt, game_memories::Array{GameMemory})
    data = game_memories_to_data(model, game_memories)
    shuffle!(data)
    function loss(x, y)
        y_policy, y_value = model(x)
        new_shape = size(y[1])[1] * size(y[1])[2]
        policy_loss = Flux.mse(reshape(y_policy, new_shape), reshape(y[1], new_shape))
        value_loss = Flux.mse(y_value, y[2])
        return policy_loss + value_loss
    end
    batch_size = 1000
    total_loss = 0.0
    data_length = length(data)
    @showprogress 1 "Training " for i in 1:batch_size:data_length
        batch = data[i:min(i+(batch_size-1), data_length)]
        Flux.train!(loss, params(model.conv_chain, model.policy_chain, model.value_chain), batch, opt)
        batch_loss = sum(loss(x, (y_policy, y_value)).data for (x, (y_policy, y_value)) in batch)
        total_loss += batch_loss
    end
    return total_loss / data_length
end

function save_model(model)
    temp_name = "model.bson." * uid()
    @save temp_name model
    mv(temp_name, "model.bson", force=true)
end

function self_play(n)
    board_size = 7
    game_memories = GameMemory[]
    optimizer = NADAM()
    while true
        bot = NNBot(board_size)
        for game in 1:n
            board = Board(board_size)
            print_board(board)
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
        game_memories_limit = 200
        game_memories = game_memories[max(1, end-(game_memories_limit-1)):end]
        @assert length(game_memories) <= game_memories_limit
        total_moves = sum(length(gm.move_memory) for gm in game_memories)
        average_game_length = total_moves / length(game_memories)
        @printf("Training on %d moves from %d games. Average moves per game: %.2f\n", total_moves, length(game_memories), average_game_length)
        loss = train(bot.model, optimizer, game_memories)
        save_model(bot.model)
        open("loss.txt", "a") do file
            println(file, (length(game_memories), average_game_length, loss))
        end
    end
end

self_play(20)
