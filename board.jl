module BoardModule

export Color, Empty, Black, White
export Point, P, other, neighbors, with_neighbors
export Board, liberties, on_board, off_board, points, valid_moves, play
export print_board, print_board_history

using Printf
using Test: @test

@enum Color::Int8 Empty Black White

function other(color::Color)
    if color == Black
        return White
    elseif color == White
        return Black
    else
        return color
    end
end

function test_other()
    @test other(Empty) == Empty
    @test other(Black) == White
    @test other(White) == Black
end

struct Point
    x::Int8
    y::Int8
end

const P = Point

function Base.:+(p1::Point, p2::Point)
    Point(p1.x + p2.x, p1.y + p2.y)
end

function neighbors(p::Point)
    [
        p + P(0, 1),
        p + P(1, 0),
        p + P(0, -1),
        p + P(-1, 0),
    ]
end

function with_neighbors(p::Point)
    [
        p,
        p + P(0, 1),
        p + P(1, 0),
        p + P(0, -1),
        p + P(-1, 0),
    ]
end

function test_point()
    @test Point(1, 2) + Point(4, -1) == Point(5, 1)
    @test P(1, 2) + P(4, -2) == P(5, 0)
    @test Set(neighbors(P(0, 0))) == Set([P(0, 1), P(1, 0), P(0, -1), P(-1, 0)])
end

mutable struct Board
    size::Int8
    positions::Array{Color}
    liberties::Array{Int16}
    history::Set{Array{Color}}
end

function Board(size)
    Board(size, fill(Empty, (size, size)), fill(0, (size, size)), Set())
end

function Base.getindex(a::AbstractArray, point::Point)
    a[point.x, point.y]
end

function Base.setindex!(a::AbstractArray, value, point::Point)
    a[point.x, point.y] = value
end

function Base.getindex(board::Board, point::Point)
    board.positions[point]
end

function Base.setindex!(board::Board, color::Color, point::Point)
    board.positions[point] = color
    update_liberties(board, with_neighbors(point))
end

function liberties(board::Board, point::Point)
    board.liberties[point]
end

function on_board(board::Board, point::Point)
    0 < point.x <= board.size && 0 < point.y <= board.size
end

function off_board(board::Board, point::Point)
    !on_board(board, point)
end

function points(board::Board)
    r = UnitRange{Int8}(Int8(1), board.size)
    map(Iterators.product(r, r)) do (x, y) P(x, y) end
end

function update_liberties(board::Board, points)
    function recurse(board::Board, this_point::Point, group::Set{Point}, group_liberties::Set{Point})
        for neighboring_point in neighbors(this_point)
            if off_board(board, neighboring_point)
                continue
            end
            if board[neighboring_point] == Empty
                push!(group_liberties, neighboring_point)
            elseif board[this_point] == board[neighboring_point] && !(neighboring_point in group)
                push!(group, neighboring_point)
                recurse(board, neighboring_point, group, group_liberties)
            end
        end
    end
    updated_liberties = fill(Int8(-1), (board.size, board.size))
    for point in points
        if off_board(board, point)
            continue
        end
        if board[point] == Empty
            board.liberties[point] = 0
            continue
        end
        if updated_liberties[point] != -1
            continue
        end
        group = Set{Point}([point])
        group_liberties = Set{Point}()
        recurse(board, point, group, group_liberties)
        for group_point in group
            updated_liberties[group_point] = length(group_liberties)
        end
    end
    for i in CartesianIndices(updated_liberties)
        if updated_liberties[i] != -1
            board.liberties[i] = updated_liberties[i]
        end
    end
end

function test_board_liberties()
    b = Board(9)
    @test b[P(1, 1)] == Empty
    b[P(1, 1)] = Black
    @test b[P(1, 1)] == Black

    b = Board(9)
    @test liberties(b, P(1, 1)) == 0
    @test liberties(b, P(1, 2)) == 0
    b[P(1, 1)] = Black
    @test liberties(b, P(1, 1)) == 2
    @test liberties(b, P(1, 2)) == 0
    b[P(1, 2)] = Black
    @test liberties(b, P(1, 1)) == 3
    @test liberties(b, P(1, 2)) == 3
    @test liberties(b, P(1, 3)) == 0
end

function remove_stones_without_liberties(board::Board, color_to_remove::Color)
    points_removed = Point[]
    for point in points(board)
        if board[point] == color_to_remove && liberties(board, point) == 0
            board.positions[point] = Empty
            push!(points_removed, point)
        end
    end
    update_liberties(board, Iterators.flatten(map(with_neighbors, points_removed)))
end

function can_place_stone(board::Board, point::Point, color::Color)
    # We can't play on an occupied point.
    if board[point] != Empty
        return false
    end
    for neighboring_point in neighbors(point)
        if off_board(board, neighboring_point)
            continue
        end
        neighboring_color = board[neighboring_point]
        # If a neighboring point is empty, then the placed stone will have a liberty.
        if neighboring_color == Empty
            return true
        end
        neighboring_liberties = liberties(board, neighboring_point)
        # We can add to one of our groups, as long as it has enough liberties.
        if neighboring_color == color && neighboring_liberties > 1
            return true
        end
        # We can take the last liberty of an opposing group.
        if neighboring_color == other(color) && neighboring_liberties == 1
            return true
        end
    end
    return false
end

function ko(board::Board, point::Point, color::Color)
    function would_be_captured(neighboring_point)
        on_board(board, neighboring_point) &&
            board[neighboring_point] == other(color) &&
            liberties(board, neighboring_point) == 1
    end
    # If no stones are going to be captured, then it is not a ko.
    if !any(would_be_captured(neighboring_point) for neighboring_point in neighbors(point))
        return false
    end
    future_board = deepcopy(board)
    play(future_board, point, color)
    return future_board.positions in board.history
end

function valid_moves(board::Board, color::Color)
    filter(points(board)) do p
        can_place_stone(board, p, color) && !ko(board, p, color)
    end
end

function play(board::Board, point::Point, color::Color)
    @assert board[point] == Empty
    push!(board.history, deepcopy(board.positions))
    @assert board.positions in board.history
    board[point] = color
    update_liberties(board, with_neighbors(point))
    remove_stones_without_liberties(board, other(color))
end

function print_board(board::Board)
    function prettify(color)
        if color == Empty
            return ' '
        elseif color == Black
            return '○'
        elseif color == White
            return '●'
        end
    end
    out = "   -" * repeat("-", board.size*2) * "--\n"
    for y in board.size:Int8(-1):1
        out *= @sprintf("%2d |", y)
        for x in Int8(1):board.size
            out *= " " * prettify(board[P(x, y)])
        end
        out *= " |\n"
    end
    out *= "   -" * repeat("-", board.size*2) * "--\n    "
    for x in Int8(1):board.size
        out *= @sprintf("%2d", x % 10)
    end
    print(out)
end

function print_board_history(board::Board)
    for positions in board.history
        b = Board(board.size)
        b.positions = positions
        print_board(b)
    end
end

function test_board_play()
    # Fill board
    b = Board(9)
    for p in points(b)
        b[p] = Black
    end
    @test b[P(1, 1)] == Black
    @test b[P(9, 9)] == Black
    @test b[P(5, 5)] == Black
    @test valid_moves(b, Black) == []
    @test valid_moves(b, White) == []

    # Atari placement
    b = Board(9)
    b[P(1, 1)] = Black
    b[P(2, 2)] = Black
    @test P(1, 2) in valid_moves(b, Black)
    @test P(1, 2) in valid_moves(b, White)
    play(b, P(1, 2), White)
    @test liberties(b, P(1, 2)) == 1

    # Ko
    b = Board(9)
    b[P(2, 1)] = Black
    b[P(1, 2)] = Black
    b[P(2, 3)] = Black
    b[P(3, 1)] = White
    b[P(4, 2)] = White
    b[P(3, 3)] = White
    @test P(3, 2) in valid_moves(b, Black)
    @test P(3, 2) in valid_moves(b, White)
    play(b, P(3, 2), Black)
    @test P(2, 2) in valid_moves(b, Black)
    @test P(2, 2) in valid_moves(b, White)
    @test b[P(3, 2)] == Black
    play(b, P(2, 2), White)
    @test b[P(3, 2)] == Empty
    @test liberties(b, P(2, 2)) == 1
    @test ko(b, P(3, 2), Black)
    @test !(P(3, 2) in valid_moves(b, Black))

    # Multiple captured stones
    b = Board(5)
    b[P(1, 1)] = Black
    b[P(2, 1)] = White
    b[P(1, 2)] = Black
    b[P(2, 2)] = White
    play(b, P(1, 3), White)
    @test b[P(1, 1)] == Empty
    @test b[P(1, 2)] == Empty
end

function test_all()
    test_other()
    test_point()
    test_board_liberties()
    test_board_play()
end

test_all()

end
