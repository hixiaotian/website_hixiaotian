### Foreword

This topic is intended only to summarize the recursive approach to binary trees.

The recursive approach seems to be concise, but in fact, there are many aspects to consider. In general, a recusive approach is designed with the following three steps in mind:

1. Base condition of recursion: Consider when to stop recursion.
2. Recursion formula: consider what is recurred and how to enter the next level of recursion.
3. Recursive return value: Consider what is returned, what type, and what value.

### Basic topic

<b>↓ Click on the title to jump directly to the leetcode title page ↓</b>

#### [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

test cases:![max-depth](https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg)

```text
Input: root = [3,9,20,null,null,15,7]
Output: 3

Input: root = [1,null,2]
Output: 2
```

Let's start with a less elaborate solution, taking into account all the circumstances:

```python
def maxDepth(self, root: Optional[TreeNode]) -> int:
    if not root: # base condition
        return 0

    if not root.left and not root.right: # base condition
        return 1

    if not root.left: # recursion formula
        return 1 + self.maxDepth(root.right)

    if not root.right: # recursion formula
        return 1 + self.maxDepth(root.left)

    # recursion formula and return value
    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

Why is this solution not refined enough? Because there are places where it can be combined. For example, the three conditions in the middle can be combined into one condition. The reason is that the base conditions of these three conditions are the same, so they can be combined into one condition:

```python
if not root.left and not root.right: # base condition
    return 1

if not root.left: # recursion formula
    return 1 + self.maxDepth(root.right)

if not root.right: # recursion formula
    return 1 + self.maxDepth(root.left)
```

In this way, a more refined solution can be obtained:

```python
def maxDepth(self, root: Optional[TreeNode]) -> int:
    if root == None: return 0

    leftMax = self.maxDepth(root.left)
    rightMax = self.maxDepth(root.right)

    return max(leftMax, rightMax) + 1
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree
- Space complexity: O (N). In the worst case, the tree is completely unbalanced, and the recursion is called N times (the height of the tree).

#### [111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)

test cases:![min-depth](https://assets.leetcode.com/uploads/2020/10/12/ex_depth.jpg)

```text
Input: root = [3,9,20,null,null,15,7]
Output: 2

Input: root = [2,null,3,null,4,null,5,null,6]
Output: 5
```

The difference between this problem and the previous one is that this problem needs to consider more situations. If a node has only one left subtree or right subtree, then Max is used to calculate the depth instead of min. So the recursive formula of this problem needs to consider more situations:

```python
def minDepth(self, root):
    if not root:
        return 0
    if None in [root.left, root.right]:
        return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
    else:
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity

#### [100. Same Tree](https://leetcode.com/problems/same-tree/)

test cases:![same-tree](https://assets.leetcode.com/uploads/2020/12/20/ex1.jpg)

```text
Input: p = [1,2,3], q = [1,2,3]
Output: true

Input: p = [1,2], q = [1,null,2]
Output: false
```

Similarly, there are many situations to consider in this question, because to judge whether two trees are the same, there are many situations to consider:

1. Both trees are empty.
2. There is a tree that is empty.
3. Neither tree is empty, but the value of the root node is different
4. Both trees are not empty and the root node has the same value

So the recursive formula of this problem is to consider the situation and the resulting code is

```python
def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if not p and not q: # base condition 1
        return True

    if not p or not q: # base condition 2
        return False

    if p.val != q.val: # base condition 3
        return False

    # recursion formula
    return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity

#### [101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

test cases:![ symmetric-tree ](https://assets.leetcode.com/uploads/2021/02/19/symtree1.jpg)

```text
Input: root = [1,2,2,3,4,4,3]
Output: true

Input: root = [1,2,2,null,3,null,3]
Output: false
```

The idea of this question is the same as that of the previous one, and there are many situations to consider. However, it should be noted that the recursive formula of this question is different, because this question is to judge whether two trees are symmetrical, so we need to use a sub-function to recursively judge the symmetry condition:

1. Both trees are empty.
2. There is a tree that is empty.
3. Neither tree is empty, but the value of the root node is different
4. The left child of the tree on the left and the right child of the tree on the right are not symmetric

So the recursive formula of this problem is to consider the situation and the resulting code is

```python
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    def _is_sym(left, right):
        if not left and not right:
            return True
        if not left and right:
            return False
        if not right and left:
            return False
        if left.val != right.val:
            return False
        left_res = _is_sym(left.left, right.right)
        right_res = _is_sym(left.right, right.left)
        return left_res and right_res

    if not root:
        return True

    return _is_sym(root.left, root.right)
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity

#### [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)

test cases:![ invert-tree ](https://assets.leetcode.com/uploads/2021/03/14/invert1-tree.jpg)

```text
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

Input: root = [2,1,3]
Output: [2,3,1]
```

This problem only needs to be modified on the original tree, but it should be noted that the replacement here needs to be stored in a temporary variable, otherwise errors will occur. Or use a python feature to directly swap the values of two variables:

```python
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return

    root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
    return root
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity

#### [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)

![ balanced-binary-tree ](https://assets.leetcode.com/uploads/2020/10/06/balance_1.jpg)

```text
Input: root = [3,9,20,null,null,15,7]
Output: true

Input: root = [1,2,2,3,3,null,null,4,4]
Output: false

Input: root = []
Output: true
```

The idea of this problem is to return the value of the current node if it is a leaf node, and to return the sum of the left or right subtree plus the value of the current node if it is not a leaf node.

```python
def isBalanced(self, root: Optional[TreeNode]) -> bool:
    if not root:
        return True

    def dfs(root):
        if not root:
            return 0

        left = dfs(root.left)
        right = dfs(root.right)

        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1

        return max(left, right) + 1

    return dfs(root) != -1
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

### Advanced type questions

#### [114. Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)

test cases:![ flatten-binary-tree-to-linked-list ](https://assets.leetcode.com/uploads/2021/01/14/flaten.jpg)

```text
Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]

Input: root = []
Output: []
```

The idea of this problem is to flatten the left subtree first, then flatten the right subtree, then put the left subtree in the position of the right subtree, and then put the original right sub-tree in the position of the right sub-tree of the rightmost node of the left sub-tree.

```python
def flatten(self, root: Optional[TreeNode]) -> None:
    """
    Do not return anything, modify root in-place instead.
    """
    if not root:
        return None

    self.flatten(root.left)
    self.flatten(root.right)

    left = root.left
    right = root.right

    root.left = None
    root.right = left

    p = root
    while p.right:
        p = p.right
    p.right = right
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

#### [116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

test cases:![ populating-next-right-pointers-in-each-node ](https://assets.leetcode.com/uploads/2019/02/14/116_sample.png)

```text
Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]

Input: root = []
Output: []
```

The idea of this problem is to first point the next pointer of the left subtree to the right subtree, then point the next pointer of the right subtree to the left subtree of the next of the parent node, and then recursively process the left and right subtrees.

```python
def connect(self, root: 'Node') -> 'Node':
    if not root:
        return None

    if root.left:
        root.left.next = root.right
        if root.next:
            root.right.next = root.next.left

    self.connect(root.left)
    self.connect(root.right)

    return root
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

#### [117. Populating Next Right Pointers in Each Node II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)

test cases:![ populating-next-right-pointers-in-each-node-ii ](https://assets.leetcode.com/uploads/2019/02/15/117_sample.png)

```text
Input: root = [1,2,3,4,5,null,7]
Output: [1,#,2,3,#,4,5,7,#]

Input: root = []
Output: []
```

The idea of this problem is the same as the above problem, but the tree of this problem is not a complete binary tree, so we need to judge whether the left subtree and the right subtree exist.

```python
def connect(self, root: 'Node') -> 'Node':
    if not root:
        return None

    if root.left:
        if root.right:
            root.left.next = root.right
        else:
            p = root.next
            while p:
                if p.left:
                    root.left.next = p.left
                    break
                elif p.right:
                    root.left.next = p.right
                    break
                p = p.next

    if root.right:
        p = root.next
        while p:
            if p.left:
                root.right.next = p.left
                break
            elif p.right:
                root.right.next = p.right
                break
            p = p.next

    self.connect(root.right)
    self.connect(root.left)

    return root
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

#### [297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

test cases:![ serialize-and-deserialize-binary-tree ](https://assets.leetcode.com/uploads/2020/09/15/serdeser.jpg)

```text
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

Input: root = []
Output: []
```

The idea of this problem is to traverse the tree in order, then save the result of the traversal into an array, then convert the array into a string, then convert the string into an array, and then convert the array into a tree.

```python
def serialize(self, root: Optional[TreeNode]) -> str:
    if not root:
        return ""

    res = []
    def preorder(root):
        if not root:
            res.append("null")
            return

        res.append(str(root.val))
        preorder(root.left)
        preorder(root.right)

    preorder(root)
    return ",".join(res)

def deserialize(self, data: str) -> Optional[TreeNode]:
    if not data:
        return None

    data = data.split(",")
    def buildTree(data):
        if not data:
            return None

        val = data.pop(0)
        if val == "null":
            return None

        root = TreeNode(int(val))
        root.left = buildTree(data)
        root.right = buildTree(data)

        return root

    return buildTree(data)
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

#### [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)

test cases:![ balanced-binary-tree ](https://assets.leetcode.com/uploads/2020/10/06/balance_1.jpg)

Input: root = [3,9,20,null,null,15,7]Output: true

![ balanced-binary-tree ](https://assets.leetcode.com/uploads/2020/10/06/balance_2.jpg)

Input: root = [1,2,2,3,3,null,null,4,4]Output: false

The idea of this problem is to traverse the tree in order, then save the result of the traversal into an array, then convert the array into a string, then convert the string into an array, and then convert the array into a tree.

```python
def isBalanced(self, root: Optional[TreeNode]) -> bool:
    def height(root):
        if not root:
            return 0

        return max(height(root.left), height(root.right)) + 1

    if not root:
        return True

    return abs(height(root.left) - height(root.right)) <= 1 and self.isBalanced(root.left) and self.isBalanced(root.right)
```

Complexity analysis:

- Time complexity: O (N ^ 2), N is the number of nodes of the tree, and considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree).
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

#### [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

test cases:![ lowest-common-ancestor-of-a-binary-tree ](https://assets.leetcode.com/uploads/2018/12/14/binarytree.png)

```python
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
```

The idea of this problem is to traverse the tree in order, then save the result of the traversal into an array, then convert the array into a string, then convert the string into an array, and then convert the array into a tree.

```python
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if not root or root == p or root == q:
        return root

    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)

    if left and right:
        return root

    return left if left else right
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

### Preorder traversal, inorder traversal, postorder traversal type problem

First, let's review the definitions of preorder traversal, inorder traversal, and postorder traversal:

- Preorder traversal: root node-> left subtree-> right subtree
- Inorder traversal: left subtree-> root node-> right subtree
- Postorder traversal: left subtree-> right subtree-> root node

One example is:

```text
    1
   / \
  2   3
 / \ / \
4  5 6  7
```

- Preorder traversal: 1 2 4 5 3 6 7
- Inorder traversal: 4 2 5 1 6 3 7
- Postorder traversal: 4 5 2 6 7 3 1

So let's take a look at this question:

#### [105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

test cases:![ construct-binary-tree-from-preorder-and-inorder-traversal ](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)

```text
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Input: preorder = [-1], inorder = [-1]
Output: [-1]
```

The key point of this problem is to focus on the first element of preorder traversal, which is the root node, and then find the root node in inorder, and then you can determine the scope of the left subtree and the right subtree, and then you can recursively build the tree.

```python

def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    if not preorder or not inorder:
        return None

    index = inorder.index(preorder.pop(0))
    root = TreeNode(inorder[index])
    root.left = self.buildTree(preorder, inorder[:index])
    root.right = self.buildTree(preorder, inorder[index+1:])

    return root
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity

#### [106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

test cases:![ construct-binary-tree-from-inorder-and-postorder-traversal ](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)

```text
Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output: [3,9,20,null,null,15,7]

Input: inorder = [-1], postorder = [-1]
Output: [-1]
```

The idea of this question is the same as that of the previous question, except that the root node of this question is the last element of postorder, and then find the root node in inorder, and then you can determine the scope of the left subtree and the right subtree, and then you can recursively build the tree.

```python
# 1. 普通做法
def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder or not postorder:
            return None

        index = inorder.index(postorder.pop())
        root = TreeNode(inorder[index])
        root.right = self.buildTree(inorder[index+1:], postorder)
        root.left = self.buildTree(inorder[:index], postorder)

        return root

# 2. 做法二： 优化空间复杂度
def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    map_inorder = {}
    for i, val in enumerate(inorder): map_inorder[val] = i
    def recur(low, high):
        if low > high: return None
        x = TreeNode(postorder.pop())
        mid = map_inorder[x.val]
        x.right = recur(mid+1, high)
        x.left = recur(low, mid-1)
        return x
    return recur(0, len(inorder)-1)
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity

#### [889. Construct Binary Tree from Preorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

test cases:![ construct-binary-tree-from-preorder-and-postorder-traversal ](https://assets.leetcode.com/uploads/2021/07/24/lc-prepost.jpg)

```text
Input: pre = [1,2,4,5,3,6,7], post = [4,5,2,6,7,3,1]
Output: [1,2,3,4,5,6,7]
```

The idea of this question is the same as that of the above two questions, except that the root node of this question is the first element of preorder, and then find the root node in postorder, and then you can determine the scope of the left subtree and the right subtree, and then you can recursively build the tree.

```python
def constructFromPrePost(self, pre: List[int], post: List[int]) -> Optional[TreeNode]:
    if not pre or not post:
        return None

    root = TreeNode(pre.pop(0))
    if len(pre) == 0:
        return root

    index = post.index(pre[0])
    root.left = self.constructFromPrePost(pre[:index+1], post[:index+1])
    root.right = self.constructFromPrePost(pre[index+1:], post[index+1:-1])

    return root
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity

### Sum type question

#### [112. Path Sum](https://leetcode.com/problems/path-sum/)

test cases:![path-sum](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

```text
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true

Input: root = [1,2,3], targetSum = 5
Output: false

Input: root = [1,2], targetSum = 0
Output: false
```

The idea of this problem is that if the current node is a leaf node, then determine whether the value of the current node is equal to targetSum, and if it is not a leaf node, then determine whether the path sum of the left or right subtree is equal to targetSum minus the value of the current node.

```python
def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
    if not root:
        return False

    if not root.left and not root.right:
        return root.val == targetSum

    return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

#### [113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)

test cases:![ path-sum-ii ](https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg)

```text
Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: [[5,4,11,2],[5,8,4,5]]

Input: root = [1,2,3], targetSum = 5
Output: []

Input: root = [1,2], targetSum = 0
Output: []
```

The idea of this problem is that if the current node is a leaf node, then determine whether the value of the current node is equal to targetSum. If it is not a leaf node, then determine whether the left subtree or the right subtree has a path sum equal to targetSum minus the value of the current node. If it exists, then add the value of current node to the path.

```python
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
    if not root:
        return []

    res = []
    def dfs(root, targetSum, path):
        if not root:
            return

        if not root.left and not root.right:
            if root.val == targetSum:
                res.append(path + [root.val])
            return

        dfs(root.left, targetSum - root.val, path + [root.val])
        dfs(root.right, targetSum - root.val, path + [root.val])

    dfs(root, targetSum, [])
    return res
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

#### [437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)

test cases:![ path-sum-iii ](https://assets.leetcode.com/uploads/2021/04/09/pathsum3-1-tree.jpg)

```text
Input: root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
Output: 3

Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: 3
```

The idea of this problem is that if the current node is a leaf node, then determine whether the value of the current node is equal to targetSum. If it is not a leaf node, then determine whether the left subtree or the right subtree has a path sum equal to targetSum minus the value of the current node. If it exists, then add the value of current node to the path.

```python
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
    if not root:
        return 0

    def dfs(root, targetSum):
        if not root:
            return 0

        res = 0
        if root.val == targetSum:
            res += 1

        res += dfs(root.left, targetSum - root.val)
        res += dfs(root.right, targetSum - root.val)

        return res

    return dfs(root, targetSum) + self.pathSum(root.left, targetSum) + self.pathSum(root.right, targetSum)
```

Complexity analysis:

- Time complexity: O (N ^ 2), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion will be called N times (the height of the tree), and each call needs to traverse the whole tree.
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

#### [666. Path Sum IV](https://leetcode.com/problems/path-sum-iv/)

To be continued

#### [129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/)

test cases:![ sum-root-to-leaf-numbers ](https://assets.leetcode.com/uploads/2021/02/19/num2tree.jpg)

```text
Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.

Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.
```

The idea of this problem is to return the value of the current node if it is a leaf node, and to return the sum of the left or right subtree plus the value of the current node if it is not a leaf node.

```python
def sumNumbers(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0

    def dfs(root, path):
        if not root:
            return 0

        if not root.left and not root.right:
            return path * 10 + root.val

        return dfs(root.left, path * 10 + root.val) + dfs(root.right, path * 10 + root.val)

    return dfs(root, 0)
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

#### [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)

The description of this problem is to find the longest path in the binary tree, which can not pass through the root node, but must pass through a node.

test cases:![ diameter-of-binary-tree ](https://assets.leetcode.com/uploads/2021/03/06/diamtree.jpg)

```text
Input: root = [1,2,3,4,5]
Output: 3

Input: root = [1,2]
Output: 1
```

The idea of this problem is to return 0 if the current node is a leaf node, and return the maximum depth of the left or right subtree plus 1 if it is not a leaf node.

```python
def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0

    self.res = 0

    def dfs(root):
        if not root:
            return 0

        left = dfs(root.left)
        right = dfs(root.right)

        self.res = max(self.res, left + right)

        return max(left, right) + 1

    dfs(root)

    return self.res
```

#### [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

The description of this problem is to find the longest path in the binary tree, which can not pass through the root node, but must pass through a node.

test cases:![ binary-tree-maximum-path-sum ](https://assets.leetcode.com/uploads/2020/10/13/exx2.jpg)

```text
Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.

Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
```

The idea of this problem is to return the value of the current node if it is a leaf node, and to return the sum of the left or right subtree plus the value of the current node if it is not a leaf node.

```python
def maxPathSum(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0

    self.res = float('-inf')

    def dfs(root):
        if not root:
            return 0

        left = max(0, dfs(root.left))
        right = max(0, dfs(root.right))

        self.res = max(self.res, left + right + root.val)

        return max(left, right) + root.val

    dfs(root)
    return self.res
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack

#### [222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/)

test cases:![ count-complete-tree-nodes ](https://assets.leetcode.com/uploads/2021/01/14/complete.jpg)

```text
Input: root = [1,2,3,4,5,6]
Output: 6

Input: root = []
Output: 0

Input: root = [1]
Output: 1
```

The idea of this problem is to return the value of the current node if it is a leaf node, and to return the sum of the left or right subtree plus the value of the current node if it is not a leaf node.

```python
def countNodes(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0

    def dfs(root):
        if not root:
            return 0

        left = dfs(root.left)
        right = dfs(root.right)

        return left + right + 1

    return dfs(root)
```

Complexity analysis:

- Time complexity: O (N), N is the number of nodes of the tree, considering that the tree is completely unbalanced in the worst case, the recursion is called N times (the height of the tree)
- Space complexity: O (N), same as time complexity, due to the space of the recursive call stack
