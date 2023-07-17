### Foreword

This topic is mainly about the algorithm of graph. In fact, the algorithm of graph is mainly to solve the problem by constructing the data structure of graph and then traversing the graph.

There are two traversal methods, one is depth-first traversal, and the other is breadth-first traversal.

In general, the time complexity of this type of problem is often O (N), because each node only needs to be traversed once.

Let's take a look at some examples, which include some of the problems we have summarized in our topic, but here we will solve them through BFS and DFS at the same time, so that we can know the difference between the two methods.

### matrix

#### [200. number of islands](https://leetcode.com/problems/number-of-islands/)

This problem is a very classic problem, and we can solve it through DFS and BFS.

test cases:

```text
input:
[
    ["1","1","1","1","0"],
    ["1","1","0","1","0"],
    ["1","1","0","0","0"],
    ["0","0","0","0","0"]
]
output: 1

input:
[
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
]

output: 3
```

DFS:

```python
class Solution:

    def numIslands(self, grid: List[List[str]]) -> int:

        def dfs(grid, i, j):
            grid[i][j] = '0'
            for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1':
                    dfs(grid, x, y)

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == '1':
                    dfs(grid, i, j)
                    count += 1

        return count
```

BFS:

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return grid

        row = len(grid)
        col = len(grid[0])
        count = 0

        def bfs(grid, i, j):
            q = collections.deque([(i, j)])
            grid[i][j] = "#" # 标记已经访问过的陆地

            while q:
                cur_x, cur_y = q.popleft()
                for x, y in (cur_x + 1, cur_y), (cur_x - 1, cur_y), (cur_x, cur_y + 1), (cur_x, cur_y - 1): # 上下左右四个方向
                    if 0 <= x < row and 0 <= y < col and grid[x][y] == "1": # 如果是陆地，那么就把它放入队列中
                        q.append((x, y))
                        grid[x][y] = "#"

        for i in range(row):
            for j in range(col):
                if grid[i][j] == "1":
                    bfs(grid, i, j)
                    count += 1

        return count
```

Complexity analysis:

- Time complexity: O (MN), where M and N are the number of rows and columns, respectively.
- Space complexity: O (MN), in the worst case, the whole grid is land, and the depth of depth-first search reaches MN.

#### [130. surrounded regions](https://leetcode.com/problems/surrounded-regions/)

The description of this topic is that given a two-dimensional matrix with a `X` sum `O` in it, we need to turn `X` the enclosed `O` into `X`.

test cases:

```text
input:
[
    ["X","X","X","X"],
    ["X","O","O","X"],
    ["X","X","O","X"],
    ["X","O","X","X"]
]
output:
[
    ["X","X","X","X"],
    ["X","X","X","X"],
    ["X","X","X","X"],
    ["X","O","X","X"]
]
```

The idea of this problem is that we first change the boundary `O` into `E`, then change the inside `O` into `X`, and finally change it `E` into `O`.

BFS:

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return board

        from itertools import product
        borders = list(
                    product(
                        range(len(board)),
                        [0, len(board[0]) - 1]
                    )) \
                + list(
                    product(
                        [0, len(board) - 1],
                        range(len(board[0]))
                    ))

        def bfs(board, i, j):
            q = collections.deque([(i, j)])

            while q:
                cur_x, cur_y = q.popleft()
                if board[cur_x][cur_y] != "O":
                    continue
                board[cur_x][cur_y] = "E"
                for x, y in (cur_x + 1, cur_y), (cur_x - 1, cur_y), (cur_x, cur_y + 1), (cur_x, cur_y - 1):
                    if 0 <= x < len(board) and 0 <= y < len(board[0]):
                            q.append((x, y))

        for i, j in borders:
            if board[i][j] != "O":
                continue
            bfs(board, i, j)

        for line in board:
            print(line)

        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == 'O':   board[r][c] = 'X'  # captured
                elif board[r][c] == 'E': board[r][c] = 'O'  # escaped
```

DFS:

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return board

        from itertools import product
        borders = list(
                    product(
                        range(len(board)),
                        [0, len(board[0]) - 1]
                    )) \
                + list(
                    product(
                        [0, len(board) - 1],
                        range(len(board[0]))
                    ))

        def dfs(board, i, j):
            if board[i][j] != "O":
                return
            board[i][j] = "E"
            for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                if 0 <= x < len(board) and 0 <= y < len(board[0]):
                    dfs(board, x, y)

        for i, j in borders:
            if board[i][j] != "O":
                continue
            dfs(board, i, j)

        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == 'O':   board[r][c] = 'X'  # captured
                elif board[r][c] == 'E': board[r][c] = 'O'  # escaped
```

Complexity analysis:

- Time complexity: O (MN), where M and N are the number of rows and columns, respectively.
- Space complexity: O (MN), in the worst case, the whole grid is land, and the depth of depth-first search reaches MN.

#### [417. pacific atlantic water flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)

The description of this topic is that given a two-dimensional matrix, the coordinates that can flow to the Pacific Ocean and the Atlantic Ocean at the same time are given.

test cases:

```text
Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
Explanation: The following cells can flow to the Pacific and Atlantic oceans, as shown below:
[0,4]: [0,4] -> Pacific Ocean
       [0,4] -> Atlantic Ocean
[1,3]: [1,3] -> [0,3] -> Pacific Ocean
       [1,3] -> [1,4] -> Atlantic Ocean
[1,4]: [1,4] -> [1,3] -> [0,3] -> Pacific Ocean
       [1,4] -> Atlantic Ocean
[2,2]: [2,2] -> [1,2] -> [0,2] -> Pacific Ocean
       [2,2] -> [2,3] -> [2,4] -> Atlantic Ocean
[3,0]: [3,0] -> Pacific Ocean
       [3,0] -> [4,0] -> Atlantic Ocean
[3,1]: [3,1] -> [3,0] -> Pacific Ocean
       [3,1] -> [4,1] -> Atlantic Ocean
[4,0]: [4,0] -> Pacific Ocean
       [4,0] -> Atlantic Ocean
Note that there are other possible paths for these cells to flow to the Pacific and Atlantic oceans.
```

The idea of this problem is to start from the boundary, find the coordinates that can flow to the Pacific Ocean and the Atlantic Ocean respectively, and then find the intersection.

BFS:

```python
class Solution:
    def bfs(self, ocean, heights):
        q = collections.deque(ocean)
        while q:
            i, j = q.popleft()
            for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                if 0 <= x < len(heights) and 0 <= y < len(heights[0]) and (x, y) not in ocean and heights[x][y] >= heights[i][j]:
                    q.append((x, y))
                    ocean.add((x, y))
        return ocean

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights: return []

        w, h = len(heights[0]), len(heights)
        pacific = set([(i, 0) for i in range(h)] + [(0, j) for j in range(w)])
        atlantic = set([(i, w - 1) for i in range(h)] + [(h - 1, j) for j in range(w)])

        return list(self.bfs(pacific, heights) & self.bfs(atlantic, heights))
```

#### [323. number of connected components in an undirected graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

The description of this topic is, given an undirected graph, how many connected parts there are.

test cases:

```text
Input: n = 5, edges = [[0,1],[1,2],[3,4]]
Output: 2
```

The idea of this problem is to use DFS or BFS to traverse the graph, then record the nodes traversed, and finally return the number of times traversed.

BFS:

```python
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        graph = collections.defaultdict(list)
        for edge in edges:
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])

        visited = set()

        def bfs(i):
            q = collections.deque([i])
            while q:
                cur = q.popleft()
                for nxt in graph[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)

        res = 0
        for i in range(n):
            if i not in visited:
                bfs(i)
                res += 1
        return res
```

DFS:

```python
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        graph = collections.defaultdict(list)
        for edge in edges:
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])

        visited = set()

        def dfs(i):
            for nxt in graph[i]:
                if nxt not in visited:
                    visited.add(nxt)
                    dfs(nxt)

        res = 0
        for i in range(n):
            if i not in visited:
                dfs(i)
                res += 1
        return res
```

Complexity analysis:

- Time complexity: $O (n + e) $
- Space complexity: $O (n + e) $

#### [261. graph valid tree](https://leetcode.com/problems/graph-valid-tree/)

The description of this topic is, given an undirected graph, to determine whether it is a tree.

test cases:

```text
Input: n = 5, and edges = [[0,1], [0,2], [0,3], [1,4]]
Output: true
```

The idea of this problem is to use DFS or BFS to traverse the graph and then determine whether there is a loop.

BFS:

```python
class Solution(object):
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """

        if len(edges) != n - 1:
            return False

        graph = [set() for _ in range(n)]
        for edge in edges:
            graph[edge[0]].add(edge[1])
            graph[edge[1]].add(edge[0])

        visited = set()
        parent = {}

        q = collections.deque([0])
        while q:
            cur = q.popleft()
            visited.add(cur)
            for nxt in graph[cur]:
                if nxt not in visited:
                    parent[nxt] = cur # 这里记录的是父节点
                    q.append(nxt)
                elif nxt != parent[cur]: # 如果当前节点已经访问过了，但是不是父节点，说明有环
                    print(parent)
                    return False

        print(parent)
        return len(visited) == n
```

At the same time, we can try to use Topological Sort to do this problem, but what we need to note here is that we need to judge whether there is a ring, if there is a ring, it is not a tree.

```python
class Solution(object):
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """

        if len(edges) != n - 1:
            return False

        graph = [set() for _ in range(n)]
        for edge in edges:
            graph[edge[0]].add(edge[1])
            graph[edge[1]].add(edge[0])

        indegrees = [0] * n
        for i in range(n):
            indegrees[i] = len(graph[i])

        queue = deque([i for i, degrees in enumerate(indegrees) if degrees == 1])
        while queue:
            node = queue.popleft()
            for child in graph[node]:
                indegrees[child] -= 1
                if indegrees[child] == 1:
                    queue.append(child)

        return all(degree == 0 for degree in indegrees)
```

#### [1926. nearest exit from entrance in maze](https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/)

The description of this topic is that given a two-dimensional matrix with a `.` sum `+`, we need to find the shortest path from the entrance to the exit.

test cases:

![ nearest_exit_from_entrance_in_maze ](https://assets.leetcode.com/uploads/2021/06/04/nearest1-grid.jpg)

```text
Input: maze = [["+","+",".","+"],[".",".",".","+"],["+","+","+","."]], entrance = [1,2]
Output: 1
Explanation: There are 3 exits in this maze at [1,0], [0,2], and [2,3].
Initially, you are at the entrance cell [1,2].
- You can reach [1,0] by moving 2 steps left.
- You can reach [0,2] by moving 1 step up.
It is impossible to reach [2,3] from the entrance.
Thus, the nearest exit is [0,2], which is 1 step away.
```

The idea of this problem is that we need to find all the exits first, and then do BFS from the entrance until we find the exit.

```python
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        q = collections.deque([(entrance[0], entrance[1], 0)])

        visited = set()
        visited.add(tuple(entrance))

        min_step = float("inf")

        def is_border(i, j):
            if entrance != [i, j]:
                return i in [0, len(maze) - 1] or j in [0, len(maze[0]) - 1]
            return False

        while q:
            i, j, step = q.popleft()
            if is_border(i, j):
                min_step = min(min_step, step)
                break

            for x, y in (i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1):
                if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and (x, y) not in visited and maze[x][y] == ".":
                    q.append((x, y, step + 1))
                    visited.add((x, y))

        return min_step if min_step != float("inf") else -1
```

Complexity analysis:

- Time complexity: O (MN), where M and N are the number of rows and columns, respectively.
- Space complexity: O (MN), in the worst case, the whole grid is land, and the depth of depth-first search reaches MN.

#### [133. clone graph](https://leetcode.com/problems/clone-graph/)

The description of this topic is that given an undirected graph, we need to clone the graph.

test cases:

![ clone_graph ](https://assets.leetcode.com/uploads/2019/11/04/133_clone_graph_question.png)

```text
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
```

The idea of this problem is that we need to use a dictionary to record the nodes that have been visited, and then use BFS or DFS to traverse the whole graph.

BFS:

```python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return node

        visited = {}

        q = collections.deque([node])
        visited[node] = Node(node.val, [])

        while q:
            cur_node = q.popleft()
            for neighbor in cur_node.neighbors:
                if neighbor not in visited:
                    visited[neighbor] = Node(neighbor.val, [])
                    q.append(neighbor)
                visited[cur_node].neighbors.append(visited[neighbor])

        return visited[node]
```

DFS:

```python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return node

        visited = {}

        def dfs(node):
            if node in visited:
                return visited[node]
            visited[node] = Node(node.val, [])
            for neighbor in node.neighbors:
                visited[node].neighbors.append(dfs(neighbor))
            return visited[node]

        return dfs(node)
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes in the graph. The depth-first search traverses the graph with each node visited only once.
- Space complexity: O (N), where N is the number of nodes in the graph. The space complexity mainly depends on the overhead of storing all nodes, which is O (N).

### Build

#### [399. evaluate division](https://leetcode.com/problems/evaluate-division/)

The description of this topic is that given some division formula, for example `a/ b = 2.0, b/ c = 3.0.`, and then given some query, for example `a/ c, b/ a, a/ e, a/ a, x/ x.`, we need to calculate the result of these queries based on the known division formula.

test cases:

```text
Input:
equations = [ ["a", "b"], ["b", "c"] ],
values = [2.0, 3.0],

queries = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ].
Output: [6.0, 0.5, -1.0, 1.0, -1.0 ]
```

The idea of this problem is that we need to use a dictionary to record the nodes that have been visited, and then use BFS or DFS to traverse the whole graph.

BFS:

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = collections.defaultdict(dict)
        for equation, value in zip(equations, values):
            src = equation[0]
            dst = equation[1]
            graph[src][dst] = value
            graph[dst][src] = 1 / value

        def bfs(graph, src, dst):
            if not graph:
                return -1

            if src not in graph:
                return -1

            q = collections.deque([(src, 1)])
            visited = set()
            visited.add(src)
            while q:
                cur = q.popleft()
                if cur[0] == dst:
                    return cur[1]

                for nxt in graph[cur[0]]:
                    if nxt not in visited:
                        q.append((nxt, cur[1] * graph[cur[0]][nxt]))
                        visited.add(nxt)
            return -1

        res = []
        for query in queries:
            res.append(bfs(graph, query[0], query[1]))
        return res
```

Complexity analysis:

- Time complexity: O (N + M), where N is the number of equations and M is the sum of the number of characters in the equations. For each query, we need O (M) time to find the corresponding edge.
- Space complexity: O (N), where N is the number of equations. We need O (N) space to store the graph.

#### [841. keys and rooms](https://leetcode.com/problems/keys-and-rooms/)

The description of this topic is that given a two-dimensional array, the elements in each array are an array, representing the keys in the room, and we need to determine whether we can enter all the rooms.

test cases:

```text
Input: [[1],[2],[3],[]]
Output: true

Input: [[1,3],[3,0,1],[2],[0]]
Output: false
```

The idea of this problem is that we need to build a graph first, store the keys in each room, and then use BFS or DFS to traverse the whole graph.

BFS:

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        graph = collections.defaultdict(set)
        for i in range(len(rooms)):
            for key in rooms[i]:
                graph[i].add(key)

        q = collections.deque([0])
        visited = set()
        visited.add(0)

        while q:
            cur = q.popleft()
            for nxt in graph[cur]:
                if nxt not in visited:
                    q.append(nxt)
                    visited.add(nxt)
        return len(visited) == len(rooms)
```

DFS:

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        graph = collections.defaultdict(set)
        for i in range(len(rooms)):
            for key in rooms[i]:
                graph[i].add(key)

        def dfs(graph, cur, visited):
            if cur in visited:
                return
            visited.add(cur)
            for nxt in graph[cur]:
                dfs(graph, nxt, visited)

        visited = set()
        dfs(graph, 0, visited)
        return len(visited) == len(rooms)
```

DFS 2 with stack:

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        graph = collections.defaultdict(set)
        for i in range(len(rooms)):
            for key in rooms[i]:
                graph[i].add(key)

        stack = [0]
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in graph[cur]:
                stack.append(nxt)
        return len(visited) == len(rooms)
```

Complexity analysis:

- Time complexity: O (N + M), where N is the number of rooms and M is the number of keys in all rooms. We need O (M) time to build the graph.
- Space complexity: O (N), where N is the number of rooms. We need O (N) space to store the graph.

#### [547. Number of Provinces](https://leetcode.com/problems/number-of-provinces/)

The description of this topic is that given a two-dimensional array of isConnected, we need to determine how many provinces there are.

test cases:

```text
Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
Output: 2

Input: isConnected = [[1,0,0],[0,1,0],[0,0,1]]
Output: 3
```

The idea of this problem is that we need to build a graph first, and then use BFS or DFS to traverse the whole graph by starting from all the nodes.

BFS:

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        graph = collections.defaultdict(set)
        for i in range(len(isConnected)):
            for j in range(len(isConnected[0])):
                if i != j and isConnected[i][j] == 1:
                    graph[i].add(j)

        q = collections.deque(graph.keys())
        visited = set()
        count = 0
        for i in range(len(isConnected)):
            if i not in visited:
                q = collections.deque([i])
                while q:
                    cur = q.popleft()

                    for nxt in graph[cur]:
                        if nxt not in visited:
                            visited.add(nxt)
                            q.append(nxt)
                count += 1
        return count
```

DFS:

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        graph = collections.defaultdict(set)
        for i in range(len(isConnected)):
            for j in range(len(isConnected[0])):
                if i != j and isConnected[i][j] == 1:
                    graph[i].add(j)

        def dfs(graph, cur, visited):
            if cur in visited:
                return
            visited.add(cur)
            for nxt in graph[cur]:
                dfs(graph, nxt, visited)

        visited = set()
        count = 0
        for i in range(len(isConnected)):
            if i not in visited:
                dfs(graph, i, visited)
                count += 1
        return count
```

#### [909. snakes and ladders](https://leetcode.com/problems/snakes-and-ladders/)

The description of this topic is to give an N X N chessboard with a number in each grid. If the number in the current grid is not -1, then we can jump to the grid where the number is located. If the number in the current grid is -1, then we can't jump to the grid. We need to judge the minimum number of steps from the upper left corner to the lower right corner.

test cases:

```text
Input: board = [[-1,-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,35,-1,-1,13,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,15,-1,-1,-1,-1]]
Output: 4

Input: board = [[-1,-1],[-1,3]]
Output: 1
```

The idea of this problem is that we need to convert the chessboard into a one-dimensional array first, and then use BFS to traverse the whole graph.

```python
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        if not board:
            return 0

        n = len(board)
        cells = [None] * (n**2)
        columns = list(range(0, n))
        label = 0
        for row in range(n - 1, -1, -1):
            for column in columns:
                cells[label] = (row, column)
                label += 1
            columns.reverse()

        dist = [float("inf")] * (n * n)
        dist[0] = 0
        q = collections.deque([0])

        def get_next(i):
            row, col = cells[i]
            return board[row][col] - 1 if board[row][col] != -1 else i

        while q:
            cur = q.popleft()

            if cur == n * n - 1:
                return dist[cur]

            for i in range(cur + 1, min(cur + 7, n * n)):
                nxt = get_next(i)
                if dist[nxt] > dist[cur] + 1:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return -1
```

Complexity analysis:

- Time complexity: O (N ^ 2), where N is the side length of the board. We need O (N ^ 2) time to build the checkerboard, and each lattice will be traversed at most once, so the total time complexity is O (N ^ 2).
- Space complexity: O (N ^ 2), where N is the side length of the board. We need O (N ^ 2) space to build the board and O (N ^ 2) space to hold the dist array.

#### [433. minimum genetic mutation](https://leetcode.com/problems/minimum-genetic-mutation/)

The description of this topic is that given a starting gene sequence, a target gene sequence, and a gene pool, we need to determine the minimum number of steps from the starting gene sequence to the target gene sequence.

test cases:

```text
Input: startGene = "AACCGGTT", endGene = "AACCGGTA", bank = ["AACCGGTA"]
Output: 1

Input: startGene = "AACCGGTT", endGene = "AAACGGTA", bank = ["AACCGGTA", "AACCGCTA", "AAACGGTA"]
Output: 2
```

The idea of this problem is that we need to use BFS to traverse the whole graph, and then use a visited array to record the sequence of genes that have been visited.

```python
class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        if not bank:
            return -1

        def can_mutate(gene):
            res = []
            for item in bank:
                diff = 0
                for i in range(len(item)):
                    if item[i] != gene[i]:
                        diff += 1
                if diff == 1:
                    res.append(item)
            return res

        bank.append(startGene)
        graph = collections.defaultdict(set)
        for item in bank:
            nxt_list = can_mutate(item)
            for nxt in nxt_list:
                graph[item].add(nxt)
        q = collections.deque()
        q.append((startGene, 0))
        visited = set()
        visited.add(startGene)

        while q:
            cur, step = q.popleft()
            if cur == endGene:
                return step
            for nxt in graph[cur]:
                if nxt not in visited:
                    q.append((nxt, step + 1))
                    visited.add(nxt)
        return -1
```

Complexity analysis:

- Time complexity: O (N ^ _ L), where N is the length of the gene pool and L is the length of the gene sequence. We need O (N ^ 2) time to build the graph, and each gene sequence will be traversed at most once, so the total time complexity is O (N ^ 2 _ 2 L).
- Space complexity: O (N ^ 2 \ * L), where N is the length of the gene pool and L is the length of the gene sequence. We need O (N ^ 2) space to build the graph and O (N ^ 2) space to hold the visited array.

#### [127. word ladder](https://leetcode.com/problems/word-ladder/)

The description of this question is, given a starting word, a target word, and a word dictionary, we need to determine the minimum number of steps from the starting word to the target word.

test cases:

```text
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5

Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: 0
```

The idea of this problem is that we need to use BFS to traverse the entire graph, and then use a visited array to record the words that have been visited.

We can try to output all the adjacent words of each word, but doing so will result in tle:

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0
        wordList.append(beginWord)
        wordList.append(endWord)
        wordList = set(wordList)
        graph = collections.defaultdict(set)
        for word in wordList: # 这样存储的话，会导致tle，因为我们嵌套了两个for loop
            for word2 in wordList:
                if word != word2:
                    diff = 0
                    for i in range(len(beginWord)):
                        if word[i] != word2[i]:
                            diff += 1
                    if diff == 1:
                        graph[word].add(word2)
                        graph[word2].add(word)
        visited = set()
        visited.add(beginWord)
        q = collections.deque([(beginWord, 1)])
        while q:
            current_word, level = q.popleft()
            for i in range(len(beginWord)):
                for word in graph[current_word]:
                    if word == endWord:
                        return level + 1
                    if word not in visited:
                        visited.add(word)
                        q.append((word, level + 1))
        return 0
```

However, we can optimize it by replacing each letter with \ *, so that the adjacent words of each word can be done in a for loop, and there will be no problem:

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0

        all_combo_dict = collections.defaultdict(list)

        for word in wordList:
            for i in range(len(beginWord)):
                all_combo_dict[word[:i] + "*" + word[i + 1:]].append(word) # 这里存储的是每一个词的每一个index的相邻词

        visited = set()
        visited.add(beginWord)

        q = collections.deque([(beginWord, 1)])

        while q:
            current_word, level = q.popleft()
            for i in range(len(beginWord)):
                temp = current_word[:i] + "*" + current_word[i+1:]

                for word in all_combo_dict[temp]:
                    if word == endWord:
                        return level + 1

                    if word not in visited:
                        visited.add(word)
                        q.append((word, level + 1))
        return 0
```

Complexity analysis:

- Time complexity: O (M ^ _ N), where M is the length of the word and N is the number of words. We need O (M ^ 2) time to build all the universal States, while each word takes O (M) time to build the universal States. In addition, we need to iterate over all the words, so the total time complexity is O (M ^ 2 _ 2 N).
- Space complexity: O (M ^ _ N), where M is the length of the word and N is the number of words. We need O (M ^ 2) space to build all the universal States while storing all the words of each universal state. In addition, we need to iterate over all the words, so the total space complexity is O (M ^ 2 _ 2 N).

### topological sort

#### [207. course schedule](https://leetcode.com/problems/course-schedule/)

The description of this topic is that given the number of courses and the prerequisites of some courses, we need to judge whether we can complete all the courses.

test cases:

```text
Input: 2, [[1,0]]
Output: true

Input: 2, [[1,0],[0,1]]
Output: false
```

The idea of this problem is to build a dictionary to record the in-degree and out-degree of each node, and then use topological sort to judge whether all the courses can be completed.

Topological sort:

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        indegree = {i: set() for i in range(numCourses)}
        outdegree = defaultdict(set)

        for u, v in prerequisites:
            indegree[u].add(v)
            outdegree[v].add(u)

        print(indegree)
        q = deque([i for i in indegree if not indegree[i]])
        res = []
        while q:
            for _ in range(len(q)):
                cur = q.popleft()
                res.append(cur)
                for nxt in outdegree[cur]:
                    indegree[nxt].remove(cur)
                    if not indegree[nxt]:
                        q.append(nxt)
        print(res)
        return True if len(res) == numCourses else False
```

Complexity analysis:

- Time complexity: O (N + M), where N is the number of courses and M is the number of prerequisites. In the process of graph construction, we need O (M) time to find the in-degree and out-degree of each node.
- Space complexity: O (N + M), where N is the number of courses and M is the number of prerequisites. We need O (M) space to store the graph.

#### [210. course schedule ii](https://leetcode.com/problems/course-schedule-ii/)

The description of this topic is that given the number of courses and the prerequisites of some courses, we need to return an order to complete all courses.

test cases:

```text
Input: 2, [[1,0]]
Output: [0,1]

Input: 4, [[1,0],[2,0],[3,1],[3,2]]
Output: [0,1,2,3] or [0,2,1,3]
```

The idea of this problem is to build a dictionary to record the in-degree and out-degree of each node, and then use topological sort to judge whether all the courses can be completed.

Topological sort:

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        indegree = {i: set() for i in range(numCourses)}
        outdegree = defaultdict(set)

        for u, v in prerequisites:
            indegree[u].add(v)
            outdegree[v].add(u)

        q = deque([i for i in indegree if not indegree[i]])
        res = []
        while q:
            for _ in range(len(q)):
                cur = q.popleft()
                res.append(cur)
                for nxt in outdegree[cur]:
                    indegree[nxt].remove(cur)
                    if not indegree[nxt]:
                        q.append(nxt)
        return res if len(res) == numCourses else []
```

Complexity analysis:

- Time complexity: O (N + M), where N is the number of courses and M is the number of prerequisites. In the process of graph construction, we need O (M) time to find the in-degree and out-degree of each node.
- Space complexity: O (N + M), where N is the number of courses and M is the number of prerequisites. We need O (M) space to store the graph.

#### [269. alien dictionary](https://leetcode.com/problems/alien-dictionary/)

The description of this topic is that given a dictionary with words in alphabetical order, we need to return an alphabetical order.

test cases:

```text
Input: [
  "wrt",
  "wrf",
  "er",
  "ett",
  "rftt"
]

Output: "wertf"
```

The idea of this problem is to find the first different letter by comparing two adjacent words, then add the order of the letter to the dictionary, and then use topological sort to judge whether all the lessons can be completed.

Topological sort:

```python
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        outdegree = collections.defaultdict(set)
        indegree = {c: set() for word in words for c in word}

        for first, second in zip(words, words[1:]):
            for c, d in zip(first, second):
                if c != d:
                    if d not in outdegree[c]:
                        outdegree[c].add(d)
                        indegree[d].add(c)
                    break
            else:
                if len(second) < len(first):
                    return ""

        res = []
        q = collections.deque([c for c in indegree if not indegree[c]])
        while q:
            cur = q.popleft()
            res.append(cur)
            for nxt in outdegree[cur]:
                indegree[nxt].remove(cur)
                if not indegree[nxt]:
                    q.append(nxt)

        if len(res) < len(indegree):
            return ""

        return "".join(res)
```

Complexity analysis:

- Time complexity: O (N + M), where N is the number of courses and M is the number of prerequisites. In the process of graph construction, we need O (M) time to find the in-degree and out-degree of each node.
- Space complexity: O (N + M), where N is the number of courses and M is the number of prerequisites. We need O (M) space to store the graph.

### Bipartite graph

#### [785. is graph bipartite](https://leetcode.com/problems/is-graph-bipartite/)

The description of this topic is that given a graph, we need to determine whether the graph is a bipartite graph.

test cases:

```text
Input: [[1,3], [0,2], [1,3], [0,2]]
Output: true

Input: [[1,2,3], [0,2], [0,1,3], [0,2]]
Output: false
```

The idea of this problem is that we can use two colors to mark each node, and then use BFS to traverse each node. If the color of the traversed node is the same as that of the current node, then it is not a bipartite graph.

BFS:

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        color = [False] * n
        visited = set()

        def bfs(vertex):
            q = collections.deque([vertex])
            visited.add(vertex)

            while q:
                cur = q.popleft()
                for nxt in graph[cur]:
                    if nxt not in visited:
                        color[nxt] = not color[cur]
                        visited.add(nxt)
                        q.append(nxt)
                    else:
                        if color[nxt] == color[cur]:
                            return False
            return True

        for vertex in range(n):
            if bfs(vertex) == False:
                return False

        return True
```

DFS:

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        color = [False] * n
        visited = [False] * n

        def dfs(vertex):
            visited[vertex] = True
            for nxt in graph[vertex]:
                if visited[nxt] == False:
                    visited[nxt] = True
                    color[nxt] = not color[vertex]
                    if dfs(nxt) == False:
                        return False
                else:
                    if color[nxt] == color[vertex]:
                        return False
            return True

        for vertex in range(n):
            if visited[vertex] == False:
                if dfs(vertex) == False:
                    return False

        return True
```

Complexity analysis:

- Time complexity: O (N + M), where N is the number of nodes and M is the number of edges in the graph. In the process of traversing the graph, we need to color each node and each edge, and the time complexity is O (1).
- Space complexity: O (N), where N is the number of nodes in the graph. We need to use O (N) space to record the color of each node, and O (N) space to record the access of each node.

#### [886. possible bipartition](https://leetcode.com/problems/possible-bipartition/)

The description of this topic is that given a graph, we need to determine whether the graph can be divided into two parts, so that the nodes in each part are not connected by edges.

test cases:

```text
Input: N = 4, dislikes = [[1,2],[1,3],[2,4]]
Output: true

Input: N = 3, dislikes = [[1,2],[1,3],[2,3]]
Output: false
```

The idea of this problem is that we can use two colors to mark each node, and then use BFS to traverse each node. If the color of the traversed node is the same as that of the current node, then it is not a bipartite graph.

BFS:

```python
class Solution:
    def possibleBipartition(self, N: int, dislikes: List[List[int]]) -> bool:
        graph = collections.defaultdict(list)
        for u, v in dislikes:
            graph[u].append(v)
            graph[v].append(u)

        color = [0] * (N + 1)
        for i in range(1, N + 1):
            if color[i] == 0:
                q = collections.deque([i])
                color[i] = 1
                while q:
                    cur = q.popleft()
                    for nxt in graph[cur]:
                        if color[nxt] == 0:
                            color[nxt] = -color[cur]
                            q.append(nxt)
                        else:
                            if color[nxt] == color[cur]:
                                return False
        return True
```

DFS:

```python
class Solution:
    def possibleBipartition(self, N: int, dislikes: List[List[int]]) -> bool:
        graph = collections.defaultdict(list)
        for u, v in dislikes:
            graph[u].append(v)
            graph[v].append(u)

        color = [0] * (N + 1)
        for i in range(1, N + 1):
            if color[i] == 0:
                color[i] = 1
                if self.dfs(i, color, graph) == False:
                    return False
        return True

    def dfs(self, vertex, color, graph):
        for nxt in graph[vertex]:
            if color[nxt] == 0:
                color[nxt] = -color[vertex]
                if self.dfs(nxt, color, graph) == False:
                    return False
            else:
                if color[nxt] == color[vertex]:
                    return False
        return True
```

Complexity analysis:

- Time complexity: O (N + M), where N is the number of nodes and M is the number of edges in the graph. In the process of traversing the graph, we need to color each node and each edge, and the time complexity is O (1).
- Space complexity: O (N), where N is the number of nodes in the graph. We need to use O (N) space to record the color of each node, and O (N) space to record the access of each node.
