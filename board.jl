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

@test other(Empty) == Empty
@test other(Black) == White
@test other(White) == Black

struct Point
    x::Int8
    y::Int8
end

P = Point

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

@test Point(1, 2) + Point(4, -1) == Point(5, 1)
@test P(1, 2) + P(4, -2) == P(5, 0)
@test Set(neighbors(P(0, 0))) == Set([P(0, 1), P(1, 0), P(0, -1), P(-1, 0)])

mutable struct Board
    size::Int8
    positions::Array{Color}
    liberties::Array{Int16}
    history::Set{Board}
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
    r = UnitRange{Int8}(1, board.size)
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

@test begin
    b = Board(9)
    @assert b[P(1, 1)] == Empty
    b[P(1, 1)] = Black
    b[P(1, 1)] == Black
end

@test begin
    b = Board(9)
    @assert liberties(b, P(1, 1)) == 0
    @assert liberties(b, P(1, 2)) == 0
    b[P(1, 1)] = Black
    @assert liberties(b, P(1, 1)) == 2
    @assert liberties(b, P(1, 2)) == 0
    b[P(1, 2)] = Black
    @assert liberties(b, P(1, 1)) == 3
    @assert liberties(b, P(1, 2)) == 3
    liberties(b, P(1, 3)) == 0
end

function remove_stones_without_liberties(board::Board, color_to_remove::Color)
    points_remove = Set()
    for point in points(board)
        if board[point] == color_to_remove && liberties(board, point) == 0
            board.positions[point] = Empty
            push!(points_removed, point)
        end
    end
    update_liberties(board, points_removed)
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
    b = deepcopy(board)
    play(b, point, color)
    return b.positions in board.history
end

function play(board::Board, point::Point, color::Color)
    @assert board[point] == Empty
    board[point] = color
    update_liberties(board, with_neighbors(point))
    remove_stones_without_liberties(other(color))
    push!(board.history, board.positions)
end






##[test]
#fn fill_board() {
#    let mut b = Board::new(19);
#    for p in b.points() {
#        b.set_position(p, Black);
#    }
#}
#
##[test]
#fn atari_placement() {
#    let mut b = Board::new(9);
#    b.set_position(P(0, 0), Black);
#    b.set_position(P(1, 1), Black);
#    assert!(b
#        .valid_moves(Black)
#        .into_iter()
#        .collect::<HashSet<Point>>()
#        .contains(&P(0, 1)));
#    assert!(b
#        .valid_moves(White)
#        .into_iter()
#        .collect::<HashSet<Point>>()
#        .contains(&P(0, 1)));
#}
#
##[test]
#fn ko_placement() {
#    let mut b = Board::new(9);
#    b.set_position(P(1, 0), Black);
#    b.set_position(P(0, 1), Black);
#    b.set_position(P(1, 2), Black);
#    b.set_position(P(2, 0), White);
#    b.set_position(P(3, 1), White);
#    b.set_position(P(2, 2), White);
#    assert!(b
#        .valid_moves(Black)
#        .into_iter()
#        .collect::<HashSet<Point>>()
#        .contains(&P(2, 1)));
#    assert!(b
#        .valid_moves(White)
#        .into_iter()
#        .collect::<HashSet<Point>>()
#        .contains(&P(2, 1)));
#    assert_eq!(b.history.len(), 0);
#    b.play(P(2, 1), Black);
#    assert_eq!(b.history.len(), 1);
#    assert!(b
#        .valid_moves(Black)
#        .into_iter()
#        .collect::<HashSet<Point>>()
#        .contains(&P(1, 1)));
#    assert!(b
#        .valid_moves(White)
#        .into_iter()
#        .collect::<HashSet<Point>>()
#        .contains(&P(1, 1)));
#    assert_eq!(b.history.len(), 1);
#    b.play(P(1, 1), White);
#    assert_eq!(b.history.len(), 2);
#    assert!(!b
#        .valid_moves(Black)
#        .into_iter()
#        .collect::<HashSet<Point>>()
#        .contains(&P(2, 1)));
#    assert!(b
#        .valid_moves(White)
#        .into_iter()
#        .collect::<HashSet<Point>>()
#        .contains(&P(2, 1)));
#}
#
##[test]
#fn valid_moves_have_liberties() {
#    let mut b = Board::new(5);
#    b.set_position(P(0, 1), Black);
#    b.set_position(P(1, 1), Black);
#    assert!(b
#        .valid_moves(White)
#        .into_iter()
#        .collect::<HashSet<Point>>()
#        .contains(&P(0, 0)));
#    b.set_position(P(1, 0), Black);
#    assert!(!b
#        .valid_moves(White)
#        .into_iter()
#        .collect::<HashSet<Point>>()
#        .contains(&P(0, 0)));
#}
#
##[test]
#fn multiple_stones_captured() {
#    let mut b = Board::new(5);
#    b.set_position(P(0, 0), Black);
#    b.set_position(P(1, 0), White);
#    b.set_position(P(0, 1), Black);
#    b.set_position(P(1, 1), White);
#    b.play(P(0, 2), White);
#    assert_eq!(b.position(P(0, 0)), Empty);
#    assert_eq!(b.position(P(0, 1)), Empty);
#}
#
##[test]
#fn board_equality() {
#    assert_eq!(Board::new(9), Board::new(9));
#    assert_ne!(Board::new(9), Board::new(19));
#    let mut b1 = Board::new(9);
#    let mut b2 = Board::new(9);
#    b1.set_position(P(0, 0), Black);
#    b2.set_position(P(0, 0), Black);
#    assert_eq!(b1, b2);
#    b2.set_position(P(0, 0), White);
#    assert_ne!(b1, b2);
#}
