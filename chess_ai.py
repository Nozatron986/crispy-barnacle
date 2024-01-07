import keras
import numpy as np
import chess
import chess.pgn
import io

def pgn_to_board(pgn_text):
    board = chess.Board()
    pgn_game = chess.pgn.read_game(io.StringIO(pgn_text))

    if pgn_game:
        board = pgn_game.board()
        for move in pgn_game.mainline_moves():
            board.push(move)

    return board

def board_to_list(board):
    board_list = []
    for rank in range(7, -1, -1):  # Iterate through ranks (rows) in reverse order
        rank_list = []
        for file in range(8):  # Iterate through files (columns)
            piece = board.piece_at(chess.square(file, rank))
            if piece:
                rank_list.append(piece.symbol())
            else:
                rank_list.append(' ')
        board_list.append(rank_list)
    return board_list

def pgn_to_boards(pgn_text):
    boards = []
    pgn_game = chess.pgn.read_game(io.StringIO(pgn_text))

    if pgn_game:
        board = chess.Board()
        boards.append(board.copy())
        for move in pgn_game.mainline_moves():
            board.push(move)
            boards.append(board.copy())

    return boards

def convert_board(board):
    fen = ""
    for row in board:
        empty_count = 0
        for piece in row:
            if piece == ' ':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                piece_opposite = piece.lower() if piece.isupper() else piece.upper()
                fen += piece_opposite
        if empty_count > 0:
            fen += str(empty_count)
        fen += '/'

    # Remove the trailing '/'
    fen = fen.rstrip('/')

    # Add default information for turn and castling
    fen += " w KQkq - 0 1"

    return chess.Board(fen)

def games_from_pgn():
    co = 0
    start = False
    game_1 = []
    c = -1
    with open('lichess_games.pgn') as f:
        for line in f:
            if co > 1000:
                break
            if co % 10000 == 0:
                print(co)
            if start:
                if line[0] != ' ' and line[0] != '[':
                    game_1[c] += line
                else:
                    start = False
                    #print(game_1[c])
            elif line[0] != ' ' and line[0] != '[':
                game_1.append('')
                game_1[c] += line
                c += 1
                start = True
            co += 1

    #chess_board = pgn_to_board(game_1[1000])
    #print(board_list)
    #print(game_1[0][-6:-3])

    all_games_dataset = []
    for i in range(len(game_1)):
        if i % 1000 == 0:
            print(i)
        game_result = game_1[i][-6:-3]
        all_boards = pgn_to_boards(game_1[i])
        board_list = []
        for i in range(len(all_boards)):
            board_list.append(board_to_list(all_boards[i]))
        result = 1 if game_result == '1-0' else 0
        dataset_game = []
        for i in board_list:
            dataset_game.append([i, result])
        all_games_dataset.append(dataset_game)
    return all_games_dataset

def splits():
    training = games_from_pgn()
    print('???')

    Xtrain = []
    Ytrain = []
    #print(training)

    for i in range(len(training)):
        if i % 1000 == 0:
            print(i)
        for j in range(len(training[i])):
            Ytrain.append(training[i][j][0])
            Xtrain.append(training[i][j][1])
    print('stage 2')
    return Xtrain, Ytrain

Ytrain, Xtrain = splits()
X_train = []

def char_to_list(c):
    if c == 'r':
        l = np.zeros(12)
        l[0] = 1
    elif c == 'n':
        l = np.zeros(12)
        l[1] = 1
    elif c == 'b':
        l = np.zeros(12)
        l[2] = 1
    elif c == 'k':
        l = np.zeros(12)
        l[3] = 1
    elif c == 'q':
        l = np.zeros(12)
        l[4] = 1
    elif c == 'p':
        l = np.zeros(12)
        l[5] = 1
    elif c == 'R':
        l = np.zeros(12)
        l[6] = 1
    elif c == 'N':
        l = np.zeros(12)
        l[7] = 1
    elif c == 'B':
        l = np.zeros(12)
        l[8] = 1
    elif c == 'K':
        l = np.zeros(12)
        l[9] = 1
    elif c == 'Q':
        l = np.zeros(12)
        l[10] = 1
    elif c == 'P':
        l = np.zeros(12)
        l[11] = 1
    else:
        l = np.zeros(12)
    return l

def board_to_training(data):
    redata = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            redata.append(char_to_list(data[i][j]))
    return redata

for i in range(len(Xtrain)):
    X_train.append([])
    X_train[i].append(board_to_training(Xtrain[i]))

print('stage 3')

model = keras.Sequential([
        keras.layers.Input(shape=(64, 12)),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dropout(.2),
        keras.layers.Dense(1, activation='sigmoid')])

for i in range(len(X_train)):
    X_train[i] = np.array(X_train[i])
Y_train = np.array(Ytrain)
for i in range(len(X_train)):
    X_train[i] = X_train[i][0]

print('stage 4')

X_train = np.array(X_train)

indices = np.arange(len(X_train))
np.random.shuffle(indices)

X_train = X_train[indices]
Y_train = Y_train[indices]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, epochs=10, validation_split=0.1)
