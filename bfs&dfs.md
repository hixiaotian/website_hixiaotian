
BFS & DFS are the most common interview questions, they can solve one same questions with both solution. Let's go!
we trained the same question both with DFS & BFS!

### MATRIX LEVEL -- BFS & DFS
##### 733. Flood Fill
1) BFS

https://assets.leetcode.com/uploads/2021/06/01/flood1-grid.jpg

Method 1: Do first then insert into queue!
```python 
import collections
def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
    if image[sr][sc] == newColor:
        return image
    
    rows, cols = len(image), len(image[0])
    oldColor = image[sr][sc]
    
    # first do! then insert base element
    image[sr][sc] = newColor
    q = deque()
    q.append([sr,sc])
    
    # loop until q cannot go further
    while q:
        i, j = q.popleft()
        
        
        # find all possible direction for this point
        for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= x < rows and 0 <= y < cols and image[x][y] == oldColor:
                # first do! then insert surronding elements
                image[x][y] = newColor
                q.append([x, y])
    return image
```
Method 2: insert into queue then do stuff!
```python 
import collections
def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
    if image[sr][sc] == newColor:
        return image
    
    rows, cols = len(image), len(image[0])
    oldColor = image[sr][sc]
    
    # insert base element first
    q = deque()
    q.append([sr,sc])
    
    # loop until q cannot go further
    while q:
        i, j = q.popleft()

        # do it here!
        image[i][j] = newColor
        
        # find all possible direction for this point
        for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= x < rows and 0 <= y < cols and image[x][y] == oldColor:
                # insert surrounding elements first!
                q.append([x, y])
    return image
```


2) DFS

matrix DFS ALWAYS use recursion, tree DFS can use DFS recursion or stack!
``` python
def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        # DFS ALWAYS start with a list of judgements for simplification
        if image[sr][sc] == newColor:
            return image
    
        rows, cols = len(image), len(image[0])
        
        # base and "always" operation
        oldColor = image[sr][sc]
        image[sr][sc] = newColor
        
        # recursion around all surronding (REMEBER To set up stop condition)
        for x, y in (sr, sc + 1), (sr + 1, sc), (sr - 1, sc), (sr, sc - 1):
            if 0 <= x < rows and 0 <= y < cols and image[x][y] == oldColor:
                self.floodFill(image, x, y, newColor)

        return image

```

##### 200. Number of Islands

1) DFS
```python
import collections
def numIslands(self, grid: List[List[str]]) -> int:
    
    rows = len(grid)
    cols = len(grid[0])
    count = 0
    
    def bfs(grid, m, n):
        res = 0
        if grid == None: return res

        q = deque()
        q.append((m, n))
        grid[m][n] = "#"
        
        while q:
            curNode = q.popleft()
            
            for x, y in (curNode[0] + 1, curNode[1]), (curNode[0], curNode[1] + 1), (curNode[0] - 1, curNode[1]), (curNode[0], curNode[1] - 1):
                if 0 <= x < rows and 0 <= y < cols and grid[x][y] == "1":
                    grid[x][y] = "#"
                    q.append((x, y))
                    
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == "1":
                bfs(grid, row, col)
                count += 1
    
    return count
```

2) BFS
```python
def numIslands(self, grid: List[List[str]]) -> int:
        
        def dfs(grid, i, j):
            if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != '1':
                return
            grid[i][j] = '#'  # THE MOST IMPORTANT STEP, VISITED STEP
            dfs(grid, i+1, j)
            dfs(grid, i-1, j)
            dfs(grid, i, j+1)
            dfs(grid, i, j-1)
        
        if not grid:
            return 0

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(grid, i, j)
                    count += 1
        return count
```






##### 490. The Maze
https://assets.leetcode.com/uploads/2021/03/31/maze1-1-grid.jpg

Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], start = [0,4], destination = [4,4]
Output: true
Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.

1) BFS
```python
import collections
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


##### 505. The Maze II
https://assets.leetcode.com/uploads/2021/03/31/maze1-1-grid.jpg
Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], start = [0,4], destination = [4,4]
Output: 12
Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.
The length of the path is 1 + 1 + 3 + 1 + 2 + 2 + 2 = 12.

1) BFS
```python
    import collections
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

        return -1 if res[destination[0]][destination[1]] == float('inf') else res[destination[0]][destination[1]]
```



### ARRAY LEVEL -- BFS & DFS

##### 78. Subsets
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

```python 
def subsets(self, nums: List[int]) -> List[List[int]]:
    sublist = []
    nums.sort()
    backtract(sublist, [], nums, 0)
    return sublist

    def backtract(sublist, templist, nums, start):
        sublist.append(templist)
        for i in range(len(nums)):
            templist.append(nums[i])
            backtract(sublist, templist, nums, i + 1)
            templist = templist[:len(templist)-2]

```




















102. Binary Tree Level Order Traversal

Why this is wrong?
```python
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    import collections
    if root == None: return res
    
    res = []
    q = deque()
    q.append(root)
    
    while q:
        temp = q.popleft()
        res.append(temp.val)
        
        if temp.left != None:
            q.append(temp.left)
            
        if temp.right != None:
            q.append(temp.right)
    
    return res
```

Use your level
```python
    import collections
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        
        res = []
        if root == None: return res
        
        q = deque()
        q.append(root)
        
        while q:
            
            # SIZE! OF! THE! QUEUE!
            size = len(q)
            
            curLevel = []
            for i in range(size):
                curNode = q.popleft()
                curLevel.append(curNode.val)
                
                if curNode.left != None:
                    q.append(curNode.left)

                if curNode.right != None:
                    q.append(curNode.right)
            
            res.append(curLevel)
        return res
```



103. Binary Tree Zigzag Level Order Traversal
```python
import collections
def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    res = []

    if root == None: return res
    
    q = deque()
    q.append(root)
    
    counter = 0
    while q:
        curLevel = []
        for i in range(len(q)):
            curNode = q.popleft()
            curLevel.append(curNode.val)
            
            if curNode.left:
                q.append(curNode.left)

            if curNode.right:
                q.append(curNode.right)
        
        if counter % 2 == 0:
            res.append(curLevel)
        else:
            res.append(curLevel[::-1])
            
        counter += 1
        
    return res

```


98. Validate Binary Search Tree

Template of inorder traversal
```python
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    if root == None: return True
    stack = []
    ls = []
    
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
            
        root = stack.pop()
        
        # condition here
        root = stack.pop()
        ls.append(root.val)
        
        root = root.right
        
    return ls
```

Inorder traversal

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        if root == None: return True
        
        stack = []
        pre = None
        
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
                
            root = stack.pop()
            
            # greater or equal than!
            if pre and pre.val >= root.val:
                return False
            
            pre = root
            root = root.right
            
        return True
```


dfs
```python

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def isValidBST(root, minValue, maxValue):
            if root == None: return True

            if root.val >= maxValue or root.val <= minValue:
                return False

            return isValidBST(root.left, minValue, root.val) and isValidBST(root.right, root.val, maxValue)

        return isValidBST(root, -sys.maxsize, sys.maxsize)  
```

230. Kth Smallest Element in a BST
```python
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    if root == None: return -1
    
    stack = []
    
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
            
        root = stack.pop()
        
        if k == 1: return root.val
        k -= 1
        
        root = root.right
        
    return -1

```

114. Flatten Binary Tree to Linked List
```python
def flatten(self, root: Optional[TreeNode]) -> None:
        def dfs(root):
            if root == None: return None
            leftLast = dfs(root.left)
            rightLast = dfs(root.right)
            
            if leftLast:
                
                ############
                leftLast.right = root.right
                root.right = root.left
                root.left = None
                ############
                
            if rightLast != None:
                return rightLast
            if leftLast != None:
                return leftLast
            
            return root
    
        dfs(root)
```

199. Binary Tree Right Side View

Why this is wrong?
```python
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    res = []
    if root == None: return res
    
    q = collections.deque()
    q.append(root)
    res.append(root.val)
    while q:
        for i in range(len(q)):
            curNode = q.popleft()
            
            if i == len(q) - 1:
                res.append(curNode.val)
            if curNode.left:
                q.append(curNode.left)
            if curNode.right:
                q.append(curNode.right)

    return res
```

It should be:

```python
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    res = []
    if root == None: return res
    
    q = collections.deque()
    q.append(root)
    while q:
        size = len(q)
        for i in range(size):
            curNode = q.popleft()
            
            if curNode.left:
                q.append(curNode.left)
                
            if curNode.right:
                q.append(curNode.right)
                
            if i == size - 1:
                res.append(curNode.val)
    return res
```

dfs
```python
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    def dfs(root, depth, res):
        if root == None: return
        
        if depth == len(res):
            res.append(root.val)
            
        dfs(root.right, depth + 1, res)
        dfs(root.left, depth + 1, res)
    
    res = []
    dfs(root, 0, res)
    return res
```



