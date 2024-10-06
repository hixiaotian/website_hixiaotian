### 前言

matrix 是一个二维数组，它的每一行都是一个一维数组，每一行的长度都相等。

在 LeetCode 中，matrix 是一个常见的数据结构，它的题型也是非常多的，比如说旋转矩阵、螺旋矩阵等等。

这些题变化多样，解题思路也是千变万化，但是我们可以总结出一些解题的套路，这些套路可以帮助我们快速解决这类问题。

让我们根据一些经典的题目，总结一下这类题目的解题套路。

### 基础题

#### [54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)

这道题的描述是给出一个二维数组，从左上角开始，按照顺时针的方向，返回所有的元素。

test case:

```text
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]
```

这道题的思路是，我们可以用一个 visited 数组来记录已经访问过的元素，然后我们按照顺时针的方向，依次访问每一个元素。具体来说，我们可以用一个 directions 数组来表示顺时针的方向，然后我们用一个 cur_direction 变量来表示当前的方向，用 i 和 j 来表示当前的位置。我们每次都先按照当前的方向，访问下一个元素，如果下一个元素不在二维数组中，或者已经访问过了，那么我们就需要改变方向。

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        visited = set()

        directions = [0, 1, 0, -1, 0] # 这里的方向是按照顺时针的方向，上右下左
        change_direction, cur_direction = 0, 0
        i, j = 0, 0
        res = [matrix[0][0]]
        visited.add((0, 0))

        while change_direction <= 1:
            while True:
                x = i + directions[cur_direction]
                y = j + directions[cur_direction + 1]

                if not (0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and (x, y) not in visited):
                    break

                change_direction = 0
                i, j = x, y
                res.append(matrix[i][j])
                visited.add((i, j))

            cur_direction = (cur_direction + 1) % 4
            change_direction += 1

        return res
```

复杂度分析：

- 时间复杂度：O(mn)，m 和 n 分别是二维数组的行数和列数。
- 空间复杂度：O(mn)，我们需要一个 visited 数组来记录已经访问过的元素。

#### [498. Diagonal Traverse](https://leetcode.com/problems/diagonal-traverse/)

这道题的描述是给出一个二维数组，从左上角开始，按照对角线的方向，返回所有的元素。

test case:

```text
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,4,7,5,3,6,8,9]
```

这道题的思路是，我们可以用一个 temp 数组来记录当前对角线上的元素，然后我们用一个 cur 变量来表示当前的对角线，用 r 和 c 来表示当前的位置。我们每次都先按照当前的方向，访问下一个元素，如果下一个元素不在二维数组中，那么我们就需要改变方向。

```python
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        if not mat or not mat[0]:
            return []

        n, m = len(mat), len(mat[0])

        result, temp = [], []

        for cur in range(n + m - 1):
            temp = []
            r = 0 if cur < m else cur - m + 1
            c = cur if cur < m else m - 1

            while r < n and c >= 0:
                temp.append(mat[r][c])
                r += 1
                c -= 1

            if cur % 2 == 0:
                result.extend(temp[::-1])
            else:
                result.extend(temp)

        return result
```

复杂度分析：

- 时间复杂度：O(mn)，m 和 n 分别是二维数组的行数和列数。
- 空间复杂度：O(mn)，我们需要一个 temp 数组来记录当前对角线上的元素。

#### [48. Rotate Image](https://leetcode.com/problems/rotate-image/)

这道题的描述是给出一个二维数组，将这个二维数组顺时针旋转 90 度。

test case:

```text
Input:
[
 [1,2,3],
 [4,5,6],
 [7,8,9]
],

Output:
[
 [7,4,1],
 [8,5,2],
 [9,6,3]
]
```

这道题的思路是，我们可以先将二维数组上下翻转，然后再按照对角线翻转。

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)

        for i in range(n // 2):
            matrix[i], matrix[n - i - 1] = matrix[n - i - 1], matrix[i]

        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

复杂度分析：

- 时间复杂度：O(mn)，m 和 n 分别是二维数组的行数和列数。
- 空间复杂度：O(1)，我们只需要常数的空间来存储一些变量。

#### [73. Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)

这道题的描述是给出一个二维数组，如果某个元素为 0，那么将这个元素所在的行和列都置为 0。

test case:

```text
Input:
[
 [1,1,1],
 [1,0,1],
 [1,1,1]
]
Output:
[
 [1,0,1],
 [0,0,0],
 [1,0,1]
]
```

这道题的思路是，我们可以用两个数组来记录哪些行和列需要置为 0，然后再遍历一遍二维数组，将需要置为 0 的行和列置为 0。

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n, m = len(matrix), len(matrix[0])

        rows, cols = [False] * n, [False] * m

        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    rows[i] = True
                    cols[j] = True

        for i in range(n):
            for j in range(m):
                if rows[i] or cols[j]:
                    matrix[i][j] = 0
```

复杂度分析：

- 时间复杂度：O(mn)，m 和 n 分别是二维数组的行数和列数。
- 空间复杂度：O(m + n)，我们需要两个数组来记录哪些行和列需要置为 0。

#### [289. Game of Life](https://leetcode.com/problems/game-of-life/)

这道题的描述是给出一个二维数组，每个元素的值只有 0 和 1，0 表示死亡，1 表示存活。如果一个元素周围有 3 个存活的元素，那么这个元素就会存活，如果一个元素周围有 2 个存活的元素，那么这个元素的状态不变，否则这个元素就会死亡。

test case:

```text
Input:
[
 [0,1,0],
 [0,0,1],
 [1,1,1],
 [0,0,0]
]

Output:
[
 [0,0,0],
 [1,0,1],
 [0,1,1],
 [0,1,0]
]
```

这道题的思路是，我们可以用一个二维数组来记录每个元素周围存活的元素的个数，然后再遍历一遍二维数组，根据题目的要求来更新每个元素的状态。具体来说，如果一个元素周围有 3 个存活的元素，那么这个元素就会存活，如果一个元素周围有 2 个存活的元素，那么这个元素的状态不变，否则这个元素就会死亡。

```python
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        neighbors = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]

        rows, cols = len(board), len(board[0])

        for row in range(rows):
            for col in range(cols):
                live_neighbors = 0

                for neighbor in neighbors:
                    r = row + neighbor[0]
                    c = col + neighbor[1]

                    if (0 <= r < rows) and (0 <= c < cols) and abs(board[r][c]) == 1:
                        live_neighbors += 1

                if board[row][col] == 1 and (live_neighbors < 2 or live_neighbors > 3):
                    # -1 signifies the cell is now dead but originally was live.
                    board[row][col] = -1

                if board[row][col] == 0 and live_neighbors == 3:
                    # 2 signifies the cell is now live but was originally dead.
                    board[row][col] = 2

        for row in range(rows):
            for col in range(cols):
                if board[row][col] > 0:
                    board[row][col] = 1
                else:
                    board[row][col] = 0
```

复杂度分析：

- 时间复杂度：O(mn)，m 和 n 分别是二维数组的行数和列数。
- 空间复杂度：O(1)，我们只需要常数的空间来存储一些变量。
