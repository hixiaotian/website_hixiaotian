### Foreword

In this topic, we will focus on two broad categories of bfs approaches: matrices and binary trees.

BFS has a classic template discussion. In python code, we can simply use queue to write its template formula:

```python
def bfs():
    q = collections.deque()
    Step 1: q put the starting point
    while q:
        cur = q.popleft()
        Step 2: deal with cur
            Step 3: q put the neighbors of cur
            q.append(neighbor)
```

Note that in the case of binary trees, there is no need to worry about the visited case of bfs, while in the case of matrices, we need to consider the visited case, otherwise it will cause an endless loop.

Next, we start with the bfs of the simplest binary tree.

### BFS for binary trees

#### BFS Template for Binary Tree

For the BFS template of binary tree, we can use queue to implement it. The specific code is as follows:

```python
def bfs(root):
    if not root:
        return
    q = collections.deque()
    q.append(root)
    while q:
        cur = q.popleft()
        # do something with cur
        if cur.left:
            q.append(cur.left)
        if cur.right:
            q.append(cur.right)
```

Let's look at some examples.

#### [112. Path Sum](https://leetcode.com/problems/path-sum/)

test cases:![image.png](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

```text
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true
Explanation: The root-to-leaf path with the target sum is shown.
```

Given a binary tree, determine whether there is a path from root to leaf such that the sum of all nodes on the path is equal to targetSum.

The idea is to use bfs to traverse all the paths, and then determine whether the sum of the paths is equal to sum. Let's use the above template to solve this problem.

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False

        q = collections.deque([(root, root.val)])

        while q:
            cur, path_sum = q.popleft()
            if not cur.left and not cur.right:
                if targetSum == path_sum:
                    return True

            elif cur.left and not cur.right:
                q.append((cur.left, path_sum + cur.left.val))

            elif not cur.left and cur.right:
                q.append((cur.right, path_sum + cur.right.val))

            else:
                q.append((cur.left, path_sum + cur.left.val))
                q.append((cur.right, path_sum + cur.right.val))

        return False
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes of the tree. Once for each node.
- Space complexity: O (N), where N is the number of nodes of the tree. The space complexity mainly depends on the overhead of the queue, and the number of elements in the queue will not exceed the number of nodes in the tree.

#### [111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)

test cases:![image.png](https://assets.leetcode.com/uploads/2020/10/12/ex_depth.jpg)

```text
Input: root = [3,9,20,null,null,15,7]
Output: 2
```

Given a binary tree, find the minimum depth from root to leaf.

The idea is to use bfs to traverse all the paths, and then determine whether the sum of the paths is equal to sum. Let's use the above template to solve this problem.

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        q = collections.deque()
        q.append(root)
        depth = 0
        while q:
            depth += 1
            size = len(q)
            for _ in range(size):
                cur = q.popleft()
                if not cur.left and not cur.right:
                    return depth
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return depth
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes of the tree. Once for each node.
- Space complexity: O (N), where N is the number of nodes of the tree. The space complexity mainly depends on the overhead of the queue, and the number of elements in the queue will not exceed the number of nodes in the tree.

#### [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

test cases:![image.png](https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg)

```text
Input: root = [3,9,20,null,null,15,7]
Output: 3
```

Given a binary tree, find the maximum depth from root to leaf.

The idea is to use bfs to traverse all the paths, and then determine whether the sum of the paths is equal to sum. Let's use the above template to solve this problem.

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        q = collections.deque()
        q.append(root)
        depth = 0
        while q:
            depth += 1
            size = len(q)
            for _ in range(size):
                cur = q.popleft()
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return depth
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes of the tree. Once for each node.
- Space complexity: O (N), where N is the number of nodes of the tree. The space complexity mainly depends on the overhead of the queue, and the number of elements in the queue will not exceed the number of nodes in the tree.

#### [129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/)

test cases:![image.png](https://assets.leetcode.com/uploads/2021/02/19/num2tree.jpg)

```text
Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.
```

Given a binary tree, find the sum of all the paths from root to leaf.

The idea is to use bfs to traverse all the paths, and then determine whether the sum of the paths is equal to sum. Let's use the above template to solve this problem.

```python
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        if not root:
            return 0
        q = collections.deque()
        q.append(root)
        res = 0
        while q:
            size = len(q)
            for _ in range(size):
                cur = q.popleft()
                if not cur.left and not cur.right:
                    res += cur.val
                if cur.left:
                    cur.left.val += cur.val * 10 # 这里是关键，每次都要乘以10
                    q.append(cur.left)
                if cur.right:
                    cur.right.val += cur.val * 10 # 这里是关键，每次都要乘以10
                    q.append(cur.right)
        return res
```

You can also process cur separately without traversing the entire queue:

```python
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        q = collections.deque([(root, str(root.val))])

        res = []
        while q:
            cur, path = q.popleft()
            if not cur.left and not cur.right:
                res.append(path)

            elif not cur.left and cur.right:
                q.append((cur.right, path + str(cur.right.val)))

            elif cur.left and not cur.right:
                q.append((cur.left, path + str(cur.left.val)))

            else:
                q.append((cur.right, path + str(cur.right.val)))
                q.append((cur.left, path + str(cur.left.val)))

        return sum([int(path) for path in res]) # 也可以不用sum，直接在上面的if not cur.left and not cur.right:里面直接res += int(path)
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes of the tree. Once for each node.
- Space complexity: O (N), where N is the number of nodes of the tree. The space complexity mainly depends on the overhead of the queue, and the number of elements in the queue will not exceed the number of nodes in the tree.

#### [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

This problem is the template problem of bfs of binary tree. We can directly use the above template to solve this problem.

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        q = collections.deque()
        q.append(root) # Step 1: q 放入root元素
        res = []
        while q:
            size = len(q) # 这里需要注意，我们需要先把当前层的size存下来，因为后面q的size会变化
            level = []
            for _ in range(size): # 使用for循环来遍历当前层的所有元素
                cur = q.popleft()  # Step 2: 处理cur的所有邻居
                level.append(cur.val)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            res.append(level) # Step 3: 把当前层的结果放入res
        return res
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes of the tree. Once for each node.
- Space complexity: O (N), where N is the number of nodes of the tree. The space complexity mainly depends on the overhead of the queue, and the number of elements in the queue will not exceed the number of nodes in the tree.

#### [107. Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)

This problem is the same as the above problem, but it is required to traverse from bottom to top. We can directly use the above template to solve this problem.

```python
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        q = collections.deque()
        q.append(root) # Step 1: q 放入root元素
        res = []
        while q:
            size = len(q) # 这里需要注意，我们需要先把当前层的size存下来，因为后面q的size会变化
            level = []
            for _ in range(size): # 使用for循环来遍历当前层的所有元素
                cur = q.popleft()  # Step 2: 处理cur的所有邻居
                level.append(cur.val)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            res.append(level) # Step 3: 把当前层的结果放入res
        return res[::-1]
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes of the tree. Once for each node.
- Space complexity: O (N), where N is the number of nodes of the tree. The space complexity mainly depends on the overhead of the queue, and the number of elements in the queue will not exceed the number of nodes in the tree.

#### [103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

test cases:![image.png](https://assets.leetcode.com/uploads/2021/02/19/tree1.jpg)

The description of this problem is that given a binary tree, return the result of the sequence traversal from left to right and from right to left.

```text
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]
```

We can directly use the above template to solve this problem, but we need to judge whether the current layer is odd or even at each layer. If it is odd, we will flip the result of the current layer.

```python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        q = collections.deque()
        q.append(root) # Step 1: q 放入root元素
        res = []
        while q:
            size = len(q) # 这里需要注意，我们需要先把当前层的size存下来，因为后面q的size会变化
            level = []
            for _ in range(size): # 使用for循环来遍历当前层的所有元素
                cur = q.popleft()  # Step 2: 处理cur的所有邻居
                level.append(cur.val)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            if len(res) % 2 == 1: # 如果是奇数层，我们就把当前层的结果翻转一下
                level = level[::-1]
            res.append(level)
        return res
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes of the tree. Once for each node.
- Space complexity: O (N), where N is the number of nodes of the tree. The space complexity mainly depends on the overhead of the queue, and the number of elements in the queue will not exceed the number of nodes in the tree.

#### [199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)

The description of this problem is, given a binary tree, return the nodes seen from the right.

test cases:![image.png](https://assets.leetcode.com/uploads/2021/02/14/tree.jpg)

```text
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]
```

We can directly use the above template to solve this problem, but we need to add the last element of the current layer to the result at each layer.

```python
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        q = collections.deque()
        q.append(root) # Step 1: q 放入root元素
        res = []
        while q:
            size = len(q) # 这里需要注意，我们需要先把当前层的size存下来，因为后面q的size会变化
            level = []
            for _ in range(size): # 使用for循环来遍历当前层的所有元素
                cur = q.popleft()  # Step 2: 处理cur的所有邻居
                level.append(cur.val)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            res.append(level[-1]) # 把当前层的最后一个元素加入到结果中
        return res
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes of the tree. Once for each node.
- Space complexity: O (N), where N is the number of nodes of the tree. The space complexity mainly depends on the overhead of the queue, and the number of elements in the queue will not exceed the number of nodes in the tree.

#### [116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

The description of this problem is that given a perfect binary tree, the nodes at each level are connected.

test cases:![image.png](https://assets.leetcode.com/uploads/2019/02/14/116_sample.png)

```text
Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
```

We can directly use the above template to solve this problem, but we need to add the last element of the current layer to the result at each layer.

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        q = collections.deque()
        q.append(root) # Step 1: q 放入root元素
        while q:
            size = len(q) # 这里需要注意，我们需要先把当前层的size存下来，因为后面q的size会变化
            for i in range(size): # 使用for循环来遍历当前层的所有元素
                cur = q.popleft()  # Step 2: 处理cur的所有邻居
                if i < size - 1: # 如果不是当前层的最后一个元素，那么就把cur的next指向q[0]
                    cur.next = q[0]
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return root
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes of the tree. Once for each node.
- Space complexity: O (N), where N is the number of nodes of the tree. The space complexity mainly depends on the overhead of the queue, and the number of elements in the queue will not exceed the number of nodes in the tree.

#### [117. Populating Next Right Pointers in Each Node II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)

The description of this problem is that given a binary tree, the nodes at each level are connected.

test cases:![image.png](https://assets.leetcode.com/uploads/2019/02/15/117_sample.png)

```text
Input: root = [1,2,3,4,5,null,7]
Output: [1,#,2,3,#,4,5,7,#]
Explanation: Given the above binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
```

We can directly use the above template to solve this problem, but we need to add the last element of the current layer to the result at each layer.

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        q = collections.deque()
        q.append(root) # Step 1: q 放入root元素
        while q:
            size = len(q) # 这里需要注意，我们需要先把当前层的size存下来，因为后面q的size会变化
            for i in range(size): # 使用for循环来遍历当前层的所有元素
                cur = q.popleft()  # Step 2: 处理cur的所有邻居
                if i < size - 1: # 如果不是当前层的最后一个元素，那么就把cur的next指向q[0]
                    cur.next = q[0]
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return root
```

Complexity analysis:

- Time complexity: O (N), where N is the number of nodes of the tree. Once for each node.
- Space complexity: O (N), where N is the number of nodes of the tree. The space complexity mainly depends on the overhead of the queue, and the number of elements in the queue will not exceed the number of nodes in the tree.

### BFS for matrices

#### [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)

The description of this problem is that given a two-dimensional array, where 1 represents land and 0 represents water, find the number of islands. To find the number of islands is to find the number of connected components.

test cases:

```text
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

We can use BFS to solve this problem. We can use each land as a starting point, then use BFS to traverse the island, mark all the land of the island as visited, and then continue to traverse the next island. Note that this is a matrix, but it may cause an endless loop, so we need to use a visited array to record whether it has been visited or not.

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return grid

        row = len(grid)
        col = len(grid[0])
        count = 0
        visited = [[0 for _ in range(col)] for _ in range(row)] # 用来标记是否访问过

        def bfs(grid, i, j):
            q = collections.deque([(i, j)])
            visited[i][j] = 1

            while q:
                cur_x, cur_y = q.popleft()
                for x, y in (cur_x + 1, cur_y), (cur_x - 1, cur_y), (cur_x, cur_y + 1), (cur_x, cur_y - 1): # 上下左右四个方向
                    if 0 <= x < row and 0 <= y < col and not visited[x][y] and grid[x][y] == "1": # 如果是陆地，那么就把它放入队列中
                        q.append((x, y))
                        visited[x][y] = 1

        for i in range(row):
            for j in range(col):
                if not visited[i][j] and grid[i][j] == "1":
                    bfs(grid, i, j)
                    count += 1

        return count
```

Of course, instead of using the visited array, we can directly use grid [I] [J] = "#" to mark the lands that have been visited.

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

- Time complexity: O (MN), where M and N are the number of rows and columns, respectively. Because we go through all the elements of the matrix at most once.
- Space complexity: O (min (M, N)), in the worst case (all land), the size of the queue can reach min (M, N).

#### [542. 01 Matrix](https://leetcode.com/problems/01-matrix/)

The description of this problem is that given a two-dimensional array, where 0 represents the ocean and 1 represents the land, find the distance between each land and the nearest ocean.

test cases:

```text
Input:
[[0,0,0],
 [0,1,0],
 [1,1,1]]

Output:
[[0,0,0],
 [0,1,0],
 [1,2,1]]
```

The idea of this problem is that we can take all the oceans as the starting point, then use BFS to traverse the land, mark the distance of the land as having been visited, and then continue to traverse the next ocean. Note that this is a matrix, but it may cause an endless loop, so we need to use a visited array to record whether it has been visited or not.

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        q = deque()
        visited = set()

        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] == 0:
                    q.append((i, j)) # 把所有的海洋都当做是起始点
                    visited.add((i, j))

        while q:
            i, j = q.popleft()

            for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                if 0 <= x < len(mat) and 0 <= y < len(mat[0]) and (x, y) not in visited:
                    mat[x][y] = mat[i][j] + 1
                    visited.add((x, y))
                    q.append((x, y))

        return mat
```

Complexity analysis:

- Time complexity: O (MN), where M and N are the number of rows and columns, respectively. Because we go through all the elements of the matrix at most once.
- Space complexity: O (MN), where M and N are the number of rows and columns, respectively. Mainly the overhead of the queue.

#### [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)

The description of this problem is that given a two-dimensional array, where 0 represents empty, 1 represents fresh oranges, and 2 represents rotten oranges, find the minimum time required to rot all oranges. If not all oranges can be rotted, then -1 is returned.

test cases:![image](https://assets.leetcode.com/uploads/2019/02/16/oranges.png)

```text
Input: [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
```

The idea of this problem is that we can take all the rotten oranges as the starting point, then use BFS to traverse the fresh oranges, mark the state of the fresh oranges as having been visited, and then continue to traverse the next rotten orange. Note that this is a matrix, but it may cause an endless loop, so we need to use a visited array to record whether it has been visited or not.

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        q = deque()
        visited = set()
        fresh = 0
        time = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 2:
                    q.append((i, j)) # 把所有的腐烂的橘子都当做是起始点
                    visited.add((i, j))
                elif grid[i][j] == 1:
                    fresh += 1

        if fresh == 0:
            return 0

        while q:
            for _ in range(len(q)):
                i, j = q.popleft()

                for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                    if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and (x, y) not in visited and grid[x][y] == 1:
                        fresh -= 1
                        visited.add((x, y))
                        q.append((x, y))
            time += 1

        return time - 1 if fresh == 0 else -1
```

Complexity analysis:

- Time complexity: O (MN), where M and N are the number of rows and columns, respectively. Because we go through all the elements of the matrix at most once.
- Space complexity: O (MN), where M and N are the number of rows and columns, respectively. Mainly the overhead of the queue.

#### [1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)

The description of this problem is that given a two-dimensional array, where 0 is empty and 1 is an obstacle, find the length of the shortest path from the upper left corner to the lower right corner. If it cannot be reached, then -1 is returned.

test cases:

```text
Input: [[0,0,0],[1,1,0],[1,1,0]]
Output: 4
```

The idea of this problem is that we can take all the emptiness as the starting point, then use BFS to traverse the obstacles, mark the state of the obstacles as having been visited, and then continue to traverse the next emptiness. Note that this is a matrix, but it may cause an endless loop, so we need to use a visited array to record whether it has been visited or not.

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        q = deque()
        visited = set()
        n = len(grid)

        if grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
            return -1

        q.append((0, 0))
        visited.add((0, 0))
        time = 0

        while q:
            for _ in range(len(q)):
                i, j = q.popleft()

                if i == n - 1 and j == n - 1:
                    return time + 1

                for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1), (i + 1, j + 1), (i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1):
                    if 0 <= x < n and 0 <= y < n and (x, y) not in visited and grid[x][y] == 0:
                        visited.add((x, y))
                        q.append((x, y))
            time += 1

        return -1
```

Complexity analysis:

- Time complexity: O (MN), where M and N are the number of rows and columns, respectively. Because we go through all the elements of the matrix at most once.
- Space complexity: O (MN), where M and N are the number of rows and columns, respectively. Mainly the overhead of the queue.

#### [1162. As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible/)

The description of this problem is that given a two-dimensional array, where 0 represents the ocean and 1 represents the land, find the farthest distance from the land in the ocean. If there is no sea or land, return -1.

test cases:

```text
Input: [[1,0,1],[0,0,0],[1,0,1]]
Output: 2
```

The idea of this problem is that we can take all the land as the starting point, then use BFS to traverse the ocean, mark the state of the ocean as having been visited, and then continue to traverse the next land. Note that this is a matrix, but it may cause an endless loop, so we need to use a visited array to record whether it has been visited or not.

```python
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        q = deque()
        visited = set()
        n = len(grid)

        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    q.append((i, j))
                    visited.add((i, j))

        if len(q) == 0 or len(q) == n * n:
            return -1

        time = 0

        while q:
            for _ in range(len(q)):
                i, j = q.popleft()

                for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                    if 0 <= x < n and 0 <= y < n and (x, y) not in visited and grid[x][y] == 0:
                        visited.add((x, y))
                        q.append((x, y))
            time += 1

        return time - 1
```

Complexity analysis:

- Time complexity: O (MN), where M and N are the number of rows and columns, respectively. Because we go through all the elements of the matrix at most once.
- Space complexity: O (MN), where M and N are the number of rows and columns, respectively. Mainly the overhead of the queue.

#### [733. Flood Fill](https://leetcode.com/problems/flood-fill/)

The description of this problem is to give a matrix to fill in a new color, and then the color will spread around from a point until it meets a color different from the original color.

test cases:![image](https://assets.leetcode.com/uploads/2021/06/01/flood1-grid.jpg)

```text
Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
Explanation: From the center of the image with position (sr, sc) = (1, 1) (i.e., the red pixel), all pixels connected by a path of the same color as the starting pixel (i.e., the blue pixels) are colored with the new color.
Note the bottom corner is not colored 2, because it is not 4-directionally connected to the starting pixel.
```

The idea of this problem is that we can use DFS or BFS to traverse all the points with the same color as the starting point, and then mark these points with a new color.

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        q = collections.deque([(sr, sc)])
        old_color = image[sr][sc]
        visited = set()
        visited.add((sr, sc))
        image[sr][sc] = color

        while q:
            x, y = q.popleft()

            for i, j in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                if 0 <= i < len(image) and 0 <= j < len(image[0]) and image[i][j] == old_color and (i, j) not in visited:
                    image[i][j] = color
                    q.append((i, j))
                    visited.add((i, j))
        return image
```

Complexity analysis:

- Time complexity: O (MN), where M and N are the number of rows and columns, respectively. Because we go through all the elements of the matrix at most once.
- Space complexity: O (MN), where M and N are the number of rows and columns, respectively. Mainly the overhead of the queue.

#### [490. The Maze](https://leetcode.com/problems/the-maze/)

The description of this problem is that given a two-dimensional array, where 0 represents the open space and 1 represents the wall, the ball can roll from the open space until it hits the wall. Find out if the ball can roll to the end.

test cases:![image](https://assets.leetcode.com/uploads/2021/03/31/maze1-1-grid.jpg)

```text
Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], start = [0,4], destination = [4,4]
Output: true
Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.
```

The idea of this problem is that we can use BFS to traverse all the points, then mark all the points as visited, stop if the wall is encountered, and return True if the end point is encountered.

```python
class Solution:
    import collections
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        rows = len(maze)
        cols = len(maze[0])

        dir = [1, 0, -1, 0, 1]
        visited = [[0 for _ in range(cols)] for _ in range(rows)]
        q = deque()
        q.append(start)
        visited[start[0]][start[1]] = 1

        while q:
            cur = q.popleft()
            if cur[0] == destination[0] and cur[1] == destination[1]:
                return True

            for i in range(4):
                newX = cur[0]
                newY = cur[1]

                while 0 <= newX < rows and 0 <= newY < cols and maze[newX][newY] != 1:
                    newX += dir[i]
                    newY += dir[i + 1]

                newX -= dir[i]
                newY -= dir[i + 1]

                if visited[newX][newY]: continue

                q.append([newX, newY])
                visited[newX][newY] = 1

        return False
```

Complexity analysis:

- Time complexity: O (MN), where M and N are the number of rows and columns, respectively. Because we go through all the elements of the matrix at most once.
- Space complexity: O (MN), where M and N are the number of rows and columns, respectively. Mainly the overhead of the queue.

#### [505. The Maze II](https://leetcode.com/problems/the-maze-ii/)

The description of this problem is that given a two-dimensional array, where 0 represents the open space and 1 represents the wall, the ball can roll from the open space until it hits the wall. Find the shortest distance between the start point and the end point of the ball.

test cases:![image](https://assets.leetcode.com/uploads/2021/03/31/maze1-1-grid.jpg)

```text
Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], start = [0,4], destination = [4,4]
Output: 12
Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.
The length of the path is 1 + 1 + 3 + 1 + 2 + 2 + 2 = 12.
```

The idea of this problem is that we can use BFS to traverse all the points, but unlike the previous problem, he needs to calculate the distance, so we need to use a two-dimensional array to record the distance from each point to the starting point, and then when traversing, if we find that the distance from the current point to the starting point is smaller than the distance recorded before, then update the distance.

```python
class Solution:
    import collections
    import sys
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:

        rows = len(maze)
        cols = len(maze[0])

        dir = [1, 0, -1, 0, 1]
        res = [[float('inf') for _ in range(cols)] for _ in range(rows)]
        q = deque()
        q.append([start[0], start[1], 0])

        while q:
            cur = q.popleft()
            if cur[2] >= res[cur[0]][cur[1]]:
                continue
            res[cur[0]][cur[1]] = cur[2]

            for i in range(4):
                newX = cur[0]
                newY = cur[1]
                path = cur[2]
                while 0 <= newX < rows and 0 <= newY < cols and maze[newX][newY] == 0:
                    newX += dir[i]
                    newY += dir[i + 1]
                    path += 1
                newX -= dir[i]
                newY -= dir[i + 1]
                path -= 1

                q.append([newX, newY, path])

        return -1 if res[destination[0]][destination[1]] == float('inf')
```
