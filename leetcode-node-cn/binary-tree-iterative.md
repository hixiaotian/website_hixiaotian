### 前言

这个专题用来总结二叉树的iterative做法，那么一般情况下iterative的做法是使用stack来实现，因为stack的特性是LIFO，所以可以用来实现DFS，而queue的特性是FIFO，所以可以用来实现BFS。

### 知识点回顾

首先我们来复习一下前序遍历，中序遍历，后序遍历的定义：

- 前序遍历：根节点 -> 左子树 -> 右子树
- 中序遍历：左子树 -> 根节点 -> 右子树
- 后序遍历：左子树 -> 右子树 -> 根节点

举一个例子来说就是：

```text
    1
   / \
  2   3
 / \ / \
4  5 6  7
```

- 前序遍历：1 2 4 5 3 6 7
- 中序遍历：4 2 5 1 6 3 7
- 后序遍历：4 5 2 6 7 3 1

### 基础类型

<b>↓点击题目就可以直接跳转到leetcode题目页面↓</b>

#### [94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

test cases:
![image.png](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```text
Input: root = [1,null,2,3]
Output: [1,3,2]

Input: root = []
Output: []

Input: root = [1]
Output: [1]
```

我们先看一下inorder recusive 的做法：

```python

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            res.append(root.val)
            dfs(root.right)
        dfs(root)
        return res
```

那么我们思考一下这个，如何把recursive转变成iterative呢？在中序遍历中，我们可以发现，我们需要先遍历左子树，然后遍历根节点，然后遍历右子树，那么我们可以先把根节点放入stack中，然后一直把左子树放入stack中，直到左子树为空，然后pop出stack中的元素，把元素的值放入res中，然后把root.right放入stack中，然后重复上面的操作。

具体操作就是：

- 先把root放入stack中
- 一直把root.left放入stack中，直到root.left为空
- 然后pop出stack中的元素，把元素的值放入res中
- 然后把root.right放入stack中，然后重复上面的操作

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = []
        while root or stack: # 这里有两个条件，一个是root != None，一个是stack != [] 
            while root: # 走到最左边
                stack.append(root)
                root = root.left
            root = stack.pop()
            res.append(root.val)
            root = root.right
        return res
```

复杂度分析：

- 时间复杂度：O(n)，因为每个节点都会被遍历一次
- 空间复杂度：O(n)，因为stack的大小最大为n

#### [144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)

test cases:
![image.png](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```text
Input: root = [1,null,2,3]
Output: [1,2,3]

Input: root = []
Output: []

Input: root = [1]
Output: [1]
```

我们先看一下preorder recusive 的做法，这个preorder的做法，跟之前的区别不大，只是在使用的时候，先把root.val放入res中，然后再遍历左子树，然后再遍历右子树。

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def dfs(root):
            if not root:
                return
            res.append(root.val)
            dfs(root.left)
            dfs(root.right)
        dfs(root)
        return res
```

那么我们思考一下这个，如何把recursive转变成iterative呢？在前序遍历中，我们可以发现，我们需要先遍历根节点，然后遍历左子树，然后遍历右子树，那么我们可以先把根节点放入stack中，然后一直把左子树放入stack中，直到左子树为空，然后pop出stack中的元素，然后把root.right放入stack中，然后重复上面的操作。

具体操作就是：

- 先把root放入stack中
- 一直把root.left放入stack中，直到root.left为空
- 然后pop出stack中的元素
- 然后把root.right放入stack中，然后重复上面的操作

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = []
        while root or stack:
            while root:
                res.append(root.val) # 因为前序遍历是先遍历根节点，所以这里先把root.val放入res中
                stack.append(root)
                root = root.left
            root = stack.pop()
            root = root.right
        return res
```

复杂度分析：

- 时间复杂度：O(n)，因为每个节点都会被遍历一次
- 空间复杂度：O(n)，因为stack的大小最大为n

#### [145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)

test cases:
![image.png](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```text
Input: root = [1,null,2,3]
Output: [3,2,1]

Input: root = []
Output: []

Input: root = [1]
Output: [1]
```

我们先看一下postorder recusive 的做法，这个postorder的做法，跟之前的区别不大，只是在使用的时候，先遍历左子树，然后再遍历右子树，然后再把root.val放入res中。

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            dfs(root.right)
            res.append(root.val)
        dfs(root)
        return res
```

那么我们思考一下这个，如何把recursive转变成iterative呢？在后序遍历中，我们可以发现，我们需要先遍历左子树，然后遍历右子树，然后遍历根节点，那么我们可以先把根节点放入stack中，然后一直把左子树放入stack中，直到左子树为空，然后pop出stack中的元素，然后把root.right放入stack中，然后重复上面的操作。

具体操作就是：

- 先把root放入stack中
- 一直把root.left放入stack中，直到root.left为空
- 然后pop出stack中的元素
- 然后把root.right放入stack中，然后重复上面的操作
- 然后把root.val放入res中

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = []
        while root or stack:
            while root:
                res.append(root.val) # 因为后序遍历是先遍历左子树，然后遍历右子树，然后遍历根节点，所以这里先把root.val放入res中
                stack.append(root)
                root = root.right
            root = stack.pop()
            root = root.left
        return res[::-1]
```

复杂度分析

- 时间复杂度：O(n)，因为每个节点都会被遍历一次
- 空间复杂度：O(n)，因为stack的大小最大为n

### 高级类型题

#### [173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)

test cases:
![image.png](https://assets.leetcode.com/uploads/2018/12/25/bst-tree.png)

这题描述太复杂了，点到题目里面去看吧。

我们直接来看，这道题怎么使用iterative的方法来做。

首先仔细思考一下，他想要的是什么，他想要的是一个iterator，这个iterator可以一直往下走，然后返回当前的值，然后再往下走，然后再返回当前的值，那么我们可以这样做：

- 首先我们需要一个stack，然后我们把root放入stack中
- 然后我们一直把root.left放入stack中，直到root.left为空
- 然后我们pop出stack中的元素，然后把root.right放入stack中，然后重复上面的操作

那我们来看一下解法和前面的有多相似：

```python
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left
            
    # @return a boolean, whether we have a next smallest number
    def hasNext(self):
        return True if len(self.stack) else False

    # @return an integer, the next smallest number
    def next(self):
        nxt = self.stack.pop()
        x = nxt.right
        while x:
            self.stack.append(x)
            x = x.left
        return nxt.val
```

复杂度分析：

- 时间复杂度：O(n)，因为每个节点都会被遍历一次
- 空间复杂度：O(n)，因为stack的大小最大为n

我们再来一道可以使用iterative 来做的题目：

#### [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

test cases:
![image.png](https://assets.leetcode.com/uploads/2020/12/01/tree2.jpg)

```text
Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.
```

这道题目，我们可以使用iterative的方法来做，我们可以使用stack来做，具体操作就是：

- 首先我们需要一个stack，然后我们把root放入stack中
- 然后我们一直把root.left放入stack中，直到root.left为空
- 然后我们pop出stack中的元素，然后把root.right放入stack中，然后重复上面的操作
- 然后我们需要一个pre，来记录上一个节点的值，然后我们每次pop出stack中的元素的时候，我们都需要判断一下，当前的值是否大于pre，如果大于，那么我们就更新pre，然后继续往下走，如果不大于，那么我们就返回False

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        stack = []
        pre = float('-inf')
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= pre:
                return False
            pre = root.val
            root = root.right
        return True
```

复杂度分析：

- 时间复杂度：O(n)，因为每个节点都会被遍历一次
- 空间复杂度：O(n)，因为stack的大小最大为n

#### [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

test cases:
![image.png](https://assets.leetcode.com/uploads/2021/01/28/kthtree1.jpg)

```text
Input: root = [3,1,4,null,2], k = 1
Output: 1
```

这道题目，我们可以使用iterative的方法来做，我们可以使用stack来做，具体操作就是：

- 首先我们需要一个stack，然后我们把root放入stack中
- 然后我们一直把root.left放入stack中，直到root.left为空
- 然后我们pop出stack中的元素，然后把root.right放入stack中，然后重复上面的操作
- 然后我们需要一个count，来记录当前是第几个元素，然后我们每次pop出stack中的元素的时候，我们都需要判断一下，当前的count是否等于k，如果等于，那么我们就返回当前的值，如果不等于，那么我们就继续往下走

```python
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack = []
        count = 0
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            count += 1
            if count == k:
                return root.val
            root = root.right
```

复杂度分析：

- 时间复杂度：O(n)，因为每个节点都会被遍历一次
- 空间复杂度：O(n)，因为stack的大小最大为n
