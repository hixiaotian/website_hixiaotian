class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word):
        cur = self.root
        for letter in word:
            cur = cur.children[letter]
        cur.is_word = True

    def is_prefix(self, target):
        cur = self.root
        for letter in target:
            cur = cur.children.get(letter)
            if cur is None:
                return False
        return True

    def is_word(self, target):
        cur = self.root
        for letter in target:
            cur = cur.children.get(letter)
            if cur is None:
                return False
        return cur.is_word


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board:
            return []

        trie = Trie()
        for word in words:
            trie.add_word(word)
        root = trie.root
        direction = [1, 0, -1, 0, 1]
        res = set()

        def backtrack(i, j, cur, path):
            if cur.is_word:
                res.add(path)

            temp = board[i][j]
            board[i][j] = "#"
            for k in range(4):
                x = i + direction[k]
                y = j + direction[k + 1]

                if 0 <= x < len(board) and 0 <= y < len(
                        board[0]) and board[x][y] in cur.children:
                    backtrack(
                        x, y, cur.children[board[x][y]],
                        path + board[x][y])
            board[i][j] = temp

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] in root.children:
                    backtrack(i, j, root.children[board[i][j]], board[i][j])

        return res
