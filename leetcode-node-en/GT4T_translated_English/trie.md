### Foreword

In this topic, we will focus on issues related to the dictionary tree (Trie).

Dictionary tree is a tree structure, which is typically used to count and sort a large number of strings (but not limited to strings), so it is often used by search engine systems for text word frequency statistics.

The utility model has the advantages that the common prefix of the character string is utilized to reduce the query time, the meaningless character string comparison is reduced to the maximum extent, and the query efficiency is higher than that of a hash table.

### Basic properties

1. The node itself does not contain a complete word;
2. From the root node to a certain node, the characters passing on the path are connected to form a character string corresponding to the node;
3. All child node paths of each node represent different characters.
4. A node can store additional information, such as frequency.
5. The node of the dictionary tree is usually not a separate data structure, but an array or hash table.

### Basic template

#### [208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

```python
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        for letter in word:
            cur = cur.children[letter]
        cur.is_word = True

    def search(self, word: str) -> bool:
        cur = self.root
        for letter in word:
            cur = cur.children.get(letter)
            if cur is None:
                return False
        return cur.is_word

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for letter in prefix:
            cur = cur.children.get(letter)
            if cur is None:
                return False
        return True
```

#### [14. Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.end = True

    def longest_prefix(self):
        res = []
        cur = self.root
        while cur:
            # return when reaches the end of word or when there are more than 1 branches
            if cur.end or len(cur.children) > 1:
                return ''.join(res)
            c = list(cur.children)[0]
            res.append(c)
            cur = cur.children[c]
        return ''.join(res)

class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ''
        T = Trie()
        for s in strs:
            T.add_word(s)
        return T.longest_prefix()
```

### Advanced questions

#### [211. Add and Search Word - Data structure design](https://leetcode.com/problems/add-and-search-word-data-structure-design/)

```python
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.root
        for letter in word:
            cur = cur.children[letter]
        cur.is_word = True

    def search(self, word: str) -> bool:
        if not word:
            return False

        cur = self.root
        return self.dfs(word, cur, 0)

    def dfs(self, word, cur, i) -> bool:
        if not cur:
            return False

        if i == len(word):
            return cur.is_word

        if word[i] == ".":
            res = []
            for child in cur.children.values():
                res.append(self.dfs(word, child, i + 1))
            return any(res)
        return self.dfs(word, cur.children.get(word[i]), i + 1)
```

#### [212. Word Search II](https://leetcode.com/problems/word-search-ii/)

```python
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word):
        cur = self.root
        for letter in word:
            cur = cur.children[letter]
        cur.is_word = True

    def search(self, word):
        cur = self.root
        for letter in word:
            cur = cur.children.get(letter)
            if cur is None:
                return False
        return cur.is_word

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board:
            return board

        def dfs(i, j, cur, part):
            if cur.is_word:
                ans.add(part)
                cur.is_word = False

            letter, board[i][j] =  board[i][j], "#"
            for x, y in (i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1):
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] in cur.children:
                    dfs(x, y, cur.children[board[x][y]], part + board[x][y])
            board[i][j] = letter

        trie = Trie()
        ans = set()
        for word in words:
            trie.add_word(word)
        word_root = trie.root
        ans = set()
        dirs = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        m, n = len(board), len(board[0])
        for r in range(m):
            for c in range(n):
                if board[r][c] in word_root.children:
                    dfs(r, c, word_root.children[board[r][c]], board[r][c])
        return ans
```
