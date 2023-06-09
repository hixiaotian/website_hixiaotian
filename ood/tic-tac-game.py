class player:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.turn_char = ""

    def update_name(self, name):
        self.name = name

    def update_char(self, char):
        self.turn_char = char

    def get_char(self):
        return self.turn_char


class Board:
    def __init__(self, size=3):
        self.size = size
        self.board = [[" " for i in range(size)] for j in range(size)]

    def get_char(self, i, j):
        return self.board[i][j]

    def put(self, i, j, char):
        self.board[i][j] = char

    def get_free_cells(self):
        free_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == " ":
                    free_cells.append((i, j))
        return free_cells

    def check_winner(self, char):
        # check rows
        for i in range(self.size):
            if all([self.board[i][j] == char for j in range(self.size)]):
                return True

        # check columns
        for j in range(self.size):
            if all([self.board[i][j] == char for i in range(self.size)]):
                return True

        # check diagonals
        if all([self.board[i][i] == char for i in range(self.size)]):
            return True

        if all([self.board[i][self.size - 1 - i] == char
                for i in range(self.size)]):
            return True

        return False

    def print_board(self):
        for i in range(self.size):
            print(self.board[i])


class TicTacToe:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.board = Board()

    def start_game(self):
        self.player1.update_char("X")
        self.player2.update_char("O")

        self.player1.update_name(input("Enter player1 name: "))
        self.player2.update_name(input("Enter player2 name: "))

        self.print_board()

        while True:
            self.player_move(self.player1)
            self.print_board()
            if self.board.check_winner(self.player1.get_char()):
                print(self.player1.name + " won the game")
                break

            self.player_move(self.player2)
            self.print_board()
            if self.board.check_winner(self.player2.get_char()):
                print(self.player2.name + " won the game")
                break

    def player_move(self, player):
        while True:
            i, j = map(int, input(player.name + " enter your move: ").split())
            if self.board.get_char(i, j) == " ":
                self.board.put(i, j, player.get_char())
                break
            else:
                print("Invalid move")

    def print_board(self):
        self.board.print_board()
