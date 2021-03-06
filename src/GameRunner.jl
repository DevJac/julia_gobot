module GameRunner

export play_game, replay_game
export move

module RandomPlayerM
export RandomPlayer, move
using BoardM
struct RandomPlayer; end
move(player::RandomPlayer, board, current_player) = rand(valid_moves(board, current_player))
end

using BoardM
using .RandomPlayerM

using Printf
using Random

struct GameState
    board :: Board
    player :: Color
    move :: Point
end

struct GameRecord
    states :: Vector{GameState}
    winner :: Color
end

function play_game(p1, p2; board_size=19, quiet=false)
    cycle_limit = board_size^2 * 2
    players = Dict{Color, Any}()
    players[Black], players[White] = shuffle([p1, p2])
    game_states = GameState[]
    board = Board(board_size)
    current_player = Black
    while length(valid_moves(board, current_player)) > 0
        m = length(game_states) > cycle_limit ? P(0, 0) : move(players[current_player], board, current_player)
        @assert m == P(0, 0) || m in valid_moves(board, current_player)
        push!(game_states, GameState(deepcopy(board), current_player, m))
        if m == P(0, 0)
            !quiet && @printf("%s resigns\n", current_player)
            break
        end
        play(board, m, current_player)
        if !quiet
            @printf("%s's (%s) move: (%d, %d)\n", current_player, players[current_player], m.x, m.y)
            print_board(board)
        end
        current_player = other(current_player)
    end
    winner = other(current_player)
    !quiet && @printf("%s wins\n", winner)
    GameRecord(game_states, winner)
end

function replay_game(game_record)
    pmove(move) = move == P(0, 0) ? "resign" : @sprintf("(%d, %d)", move.x, move.y)
    for game_state in game_record.states
        print_board(game_state.board)
        @printf("%s's move: %s\n", game_state.player, pmove(game_state.move))
        next_board = deepcopy(game_state.board)
        play(next_board, game_state.move, game_state.player)
        next_player = other(game_state.player)
        if length(valid_moves(next_board, next_player)) == 0
            print_board(next_board)
            @printf("%s has no valid moves\n", next_player)
        end
    end
    @printf("%s wins\n", game_record.winner)
end

function test_all()
    play_game(RandomPlayer(), RandomPlayer(), board_size=9)
end

end # module
