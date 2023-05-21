BFS & DFS are the most common interview questions, they can solve one same questions with both solution. Let's go!
we trained the same question both with DFS & BFS!

### MATRIX LEVEL -- BFS & DFS
##### 733. Flood Fill

1) BFS
![flood1-grid](https://hi-elliot.com/content/images/2021/12/flood1-grid.jpeg)

* Bullet Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2
* Bullet Output: [[2,2,2],[2,2,0],[2,0,1]]
* Bullet Explanation: From the center of the image with position (sr, sc) = (1, 1) (i.e., the red pixel), all pixels connected by a path of the same color as the starting pixel (i.e., the blue pixels) are colored with the new color.
Note the bottom corner is not colored 2, because it is not 4-directionally connected to the starting pixel.

Method 1: Do the task first then insert into queue!
```python 
import collections
def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
    if image[sr][sc] == newColor:
        return image
    
    rows, cols = len(image), len(image[0])
    oldColor = image[sr][sc]
    
    # first do the task! then insert base element
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
* Bullet Input: maze = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]], start = [0,4], destination = [4,4]
* Bullet Output: 12
* Bullet Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.
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

Solution 1

Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

```python 
def subsets(self, nums: List[int]) -> List[List[int]]:
    def backtract(sublist, templist, nums, start):
        sublist.append(templist[:])
        for i in range(start, len(nums)):
            templist.append(nums[i])
            backtract(sublist, templist, nums, i + 1)
            templist.pop()
        return sublist
            
    sublist = []
    nums.sort()
    return backtract(sublist, [], nums, 0)
```
Solution 2
```python 
def subsets(self, nums):
    result = [[]]
    for num in nums:
        result += [i + [num] for i in result]
    return result
```

##### 90. Subsets II
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

```python 
def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
    def backtract(sublist, templist, nums, start):
        sublist.append(templist[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            templist.append(nums[i])
            backtract(sublist, templist, nums, i + 1)
            templist.pop()
        return sublist
            
    sublist = []
    nums.sort()
    return backtract(sublist, [], nums, 0)
```

##### 46. Permutations

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

```python
def backtract(self, reslist, templist, nums) -> List[List[int]]:
        if len(templist) == len(nums):
            reslist.append(templist[:])
        else:
            for i in range(len(nums)):
                if nums[i] in templist:
                    continue
                templist.append(nums[i])
                self.backtract(reslist, templist, nums)
                templist.pop()
        return reslist
    
def permute(self, nums: List[int]) -> List[List[int]]:
    reslist = []
    nums.sort()
    return self.backtract(reslist, [], nums)
```

##### 47. Permutations II

Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]

```python
def backtract(self, reslist, templist, nums, used) -> List[List[int]]:
        if len(templist) == len(nums):
            reslist.append(templist[:])
        else:
            for i in range(len(nums)):
                if used[i] == 1 or i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                    continue
                used[i] = 1
                templist.append(nums[i])
                self.backtract(reslist, templist, nums, used)
                used[i] = 0
                templist.pop()
        return reslist
    
def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    reslist = []
    boolean = [0 for _ in range(len(nums))]
    nums.sort()
    return self.backtract(reslist, [], nums, boolean)
```


##### 39. Combination Sum
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.

```python
def backtrack(self, reslist, templist, nums, remain, start) -> List[List[int]]:
        if remain < 0: 
            return
        elif remain == 0:
            reslist.append(templist[:])
        else:
            for i in range(start, len(nums)):
                templist.append(nums[i])
                self.backtrack(reslist, templist, nums, remain - nums[i], i)
                templist.pop()
        return reslist
    
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        reslist = []
        candidates.sort()
        return self.backtrack(reslist, [], candidates, target, 0)
```


##### 40. Combination Sum II
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]

```python
def backtrack(self, reslist, templist, nums, remain, start) -> List[List[int]]:
        if remain < 0: 
            return
        elif remain == 0:
            reslist.append(templist[:])
        else:
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                templist.append(nums[i])
                self.backtrack(reslist, templist, nums, remain - nums[i], i + 1)
                templist.pop()
        return reslist
    
def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    reslist = []
    candidates.sort()
    return self.backtrack(reslist, [], candidates, target, 0)
        
```

##### 131. Palindrome Partitioning

Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]

https://leetcode.com/problems/palindrome-partitioning/Figures/131/time_complexity.png

```python
    def isPalindrome(self, s, start, end) -> bool:
        temp = s[start:end + 1]
        return temp == temp[::-1]
    
    def backtrack(self, reslist, templist, s, start) -> List[List[int]]:
        if start == len(s):
            reslist.append(templist[:])
        else:
            for i in range(start, len(s)):
                if self.isPalindrome(s, start, i):
                    templist.append(s[start: i + 1])
                    self.backtrack(reslist, templist, s, i + 1)
                    templist.pop()
                    
        return reslist
    
    def partition(self, s: str) -> List[List[str]]:
        reslist = []
        return self.backtrack(reslist, [], s, 0)
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

### BINARY TREE LEVEL -- BFS & DFS

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
