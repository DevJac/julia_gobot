include("board.jl")

using .BoardModule

using Test: @test
using Flux

@test begin
    b = Board(9)
    b[P(1, 1)] = Black
    b[P(1, 1)] == Black
end

function encode_board(board::Board, color::Color)
    valid_moves_set = Set(valid_moves(board, color))
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
        t[p.x, p.y, 11] = p in valid_moves_set
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
            softmax)
        value_chain = Chain(
            value_hidden_layer,
            value_output_layer)
        conv_chain_output = conv_chain(x)
        return (policy_chain(conv_chain_output), value_chain(conv_chain_output))
    end
end

@test begin
    b = Board(9)
    m = create_model(b)
    y_policy, y_value = m(encode_board(b, Black))
    @assert size(y_policy) == (Int16(b.size)^2, 1)
    size(y_value) == (1, 1)
end
