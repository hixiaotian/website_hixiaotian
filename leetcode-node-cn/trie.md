### 前言

在这个专题，我们将主要讨论字典树（Trie）的相关问题。

字典树是一种树形结构，典型应用是用于统计和排序大量的字符串（但不仅限于字符串），所以经常被搜索引擎系统用于文本词频统计。

它的优点是：利用字符串的公共前缀来减少查询时间，最大限度地减少无谓的字符串比较，查询效率比哈希表高。

### 基本性质

1. 结点本身不存完整单词；
2. 从根结点到某一结点，路径上经过的字符连接起来，为该结点对应的字符串；
3. 每个结点的所有子结点路径代表的字符都不相同。
4. 结点可以存储额外信息，比如频次。
5. 字典树的结点一般不是单独的数据结构，而是一个数组或者哈希表。

### 基本模板

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

### 进阶题目

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
