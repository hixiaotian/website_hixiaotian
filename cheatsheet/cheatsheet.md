1. Reverse Linkedlist

```python
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev

def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head
    p = self.reverseList(head.next)
    head.next.next = head
    head.next = None
    return p
```

2. Remove Element

```python
def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy_head = ListNode(-1)
        dummy_head.next = head
        prev, cur = dummy_head, head

        while cur:
            if cur.val == val:
                prev.next = cur.next
            else:
                prev = prev.next
            cur = cur.next
        return dummy_head.next
```

3. Calculator

```python
def calculate(self, s):
    def update(op, v):
        if op == "+": stack.append(v)
        if op == "-": stack.append(-v)
        if op == "*": stack.append(stack.pop() * v)           #for BC II and BC III
        if op == "/": stack.append(int(stack.pop() / v))      #for BC II and BC III

    it, num, stack, sign = 0, 0, [], "+"

    while it < len(s):
        if s[it].isdigit():
            num = num * 10 + int(s[it])
        elif s[it] in "+-*/":
            update(sign, num)
            num, sign = 0, s[it]
        elif s[it] == "(":                                        # For BC I and BC III
            num, j = self.calculate(s[it + 1:])
            it = it + j
        elif s[it] == ")":                                        # For BC I and BC III
            update(sign, num)
            return sum(stack), it + 1
        it += 1
    update(sign, num)
    return sum(stack)
```

4. 二分法

```python
def search(self, nums: List[int], target: int) -> int:
    if not nums:
        return -1

    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        if target < nums[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return -1
```

5. binary tree iterative

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            res.append(root.val)
            root = root.right
        return res
```
