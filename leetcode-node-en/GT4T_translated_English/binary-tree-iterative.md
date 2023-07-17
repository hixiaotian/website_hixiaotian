### Foreword

This topic is used to summarize the iterative method of binary tree. In general, the iterative method is implemented by using stack. Because the feature of stack is LIFO, it can be used to implement DFS, while the feature of queue is FIFO, it can be used to implement BFS.

### Knowledge point review

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

### Foundation type

<b>↓ Click on the title to jump directly to the leetcode title page ↓</b>

#### [94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

test cases:![image.png](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```text
Input: root = [1,null,2,3]
Output: [1,3,2]

Input: root = []
Output: []

Input: root = [1]
Output: [1]
```

Let's take a look at how inorder recusive works:

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

So let's think about this, how do we turn recursive into iterative? In the inorder traversal, we can find that we need to traverse the left subtree first, then traverse the root node, and then traverse the right subtree, so we can put the root node into the stack first, then put the left subtree into the stack until the left sub-tree is empty, then pop out the elements in the stack, and put the values of the elements into res. Then put the root. Right into the stack and repeat the above operation.

The specific operation is:

- Put root in the stack first.
- Keep putting the root. Left into the stack until the root. Left is empty
- Then pop out the element in the stack and put the value of the element into res.
- Then put the root. Right into the stack and repeat the above operation.

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

Complexity analysis:

- Time complexity: O (n), because each node is traversed once
- Space complexity: O (n), because the size of the stack is at most n

#### [144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)

test cases:![image.png](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```text
Input: root = [1,null,2,3]
Output: [1,2,3]

Input: root = []
Output: []

Input: root = [1]
Output: [1]
```

Let's take a look at the preorder recusive method. This preorder method is not very different from the previous one, except that when it is used, the root. Val is put into res first, and then the left subtree is traversed, and then the right subtree is traversed.

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

So let's think about this, how do we turn recursive into iterative? In the preorder traversal, we can find that we need to traverse the root node first, then traverse the left subtree, and then traverse the right subtree, so we can put the root node into the stack first, and then put the left subtree into the stack until the left sub-tree is empty, and then pop out the elements in the stack. Then put the root. Right into the stack and repeat the above operation.

The specific operation is:

- Put root in the stack first.
- Keep putting the root. Left into the stack until the root. Left is empty
- Then pop out the elements in the stack.
- Then put the root. Right into the stack and repeat the above operation.

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

Complexity analysis:

- Time complexity: O (n), because each node is traversed once
- Space complexity: O (n), because the size of the stack is at most n

#### [145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)

test cases:![image.png](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```text
Input: root = [1,null,2,3]
Output: [3,2,1]

Input: root = []
Output: []

Input: root = [1]
Output: [1]
```

Let's take a look at the postorder recusive method. This postorder method is not very different from the previous one, except that when it is used, the left subtree is traversed first, then the right subtree is traversed, and then the root. Val is put into res.

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

So let's think about this, how do we turn recursive into iterative? In the post-order traversal, we can find that we need to traverse the left subtree first, then traverse the right subtree, and then traverse the root node, so we can put the root node into the stack first, then put the left subtree into the stack until the left sub-tree is empty, and then pop out the elements in the stack. Then put the root. Right into the stack and repeat the above operation.

The specific operation is:

- Put root in the stack first.
- Keep putting the root. Left into the stack until the root. Left is empty
- Then pop out the elements in the stack.
- Then put the root. Right into the stack and repeat the above operation.
- Then put the root. Val into res.

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

Complexity analysis

- Time complexity: O (n), because each node is traversed once
- Space complexity: O (n), because the size of the stack is at most n

### Advanced type questions

#### [173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)

test cases:![image.png](https://assets.leetcode.com/uploads/2018/12/25/bst-tree.png)

The description of this question is too complicated. Click on the title to see it.

Let's look directly at how to use the iterative method to solve this problem.

First, think carefully about what he wants. What he wants is an iterator that can go all the way down, then return the current value, then go down again, and then return the current value. So we can do this:

- First we need a stack, and then we put root in the stack.
- Then we keep putting the root. Left in the stack until the root. Left is empty.
- Then we pop out the elements in the stack, then put the root. Right into the stack, and then repeat the above operation.

So let's see how similar the solution is to the previous one:

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

Complexity analysis:

- Time complexity: O (n), because each node is traversed once
- Space complexity: O (n), because the size of the stack is at most n

Let's take another question that can be done with iterative:

#### [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

test cases:![image.png](https://assets.leetcode.com/uploads/2020/12/01/tree2.jpg)

```text
Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.
```

For this problem, we can use the iterative method to do it, and we can use stack to do it. The specific operation is:

- First we need a stack, and then we put root in the stack.
- Then we keep putting the root. Left in the stack until the root. Left is empty.
- Then we pop out the elements in the stack, then put the root. Right into the stack, and then repeat the above operation.
- Then we need a pre to record the value of the previous node, and then every time we pop out the elements in the stack, we need to determine whether the current value is greater than pre, if so, then we update pre, and then continue to go down, if not, then we return False.

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

Complexity analysis:

- Time complexity: O (n), because each node is traversed once
- Space complexity: O (n), because the size of the stack is at most n

#### [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

test cases:![image.png](https://assets.leetcode.com/uploads/2021/01/28/kthtree1.jpg)

```text
Input: root = [3,1,4,null,2], k = 1
Output: 1
```

For this problem, we can use the iterative method to do it, and we can use stack to do it. The specific operation is:

- First we need a stack, and then we put root in the stack.
- Then we keep putting the root. Left in the stack until the root. Left is empty.
- Then we pop out the elements in the stack, then put the root. Right into the stack, and then repeat the above operation.
- Then we need a count to record the current number of elements, and then every time we pop out the elements in the stack, we need to determine whether the current count is equal to K, if so, then we return the current value, if not, then we continue to go down.

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

Complexity analysis:

- Time complexity: O (n), because each node is traversed once
- Space complexity: O (n), because the size of the stack is at most n
