### 前言

在这个专题，我们重点讨论链表相关的问题。

其实在很多的链表问题当中，我们都可以先建立一个 dummy node，这样做的好处是我们不需要对头节点进行特殊的处理，而且在最后返回的时候也比较方便。

比如下面这道题，我们需要对链表进行反转，就可以先建立一个 dummy node，然后不断地把 cur.next 指向 dummy.next，然后把 dummy.next 指向 cur.next，这样就可以完成反转了。

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        cur = head
        while cur and cur.next:
            tmp = cur.next
            cur.next = tmp.next
            tmp.next = dummy.next
            dummy.next = tmp
        return dummy.next
```

### [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

这道题的描述是反转一个链表，我们可以使用上面的方法来解决。

test cases:

```text
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Input: head = [1,2]
Output: [2,1]

Input: head = []
Output: []
```

这道题的做法是，我们使用一个 dummy node，然后不断地把 cur.next 指向 dummy.next，然后把 dummy.next 指向 cur.next，这样就可以完成反转了。

我们可以简单用一个例子来说明这个过程：

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
dummy -> 2 -> 1 -> 3 -> 4 -> 5
dummy -> 3 -> 2 -> 1 -> 4 -> 5
dummy -> 4 -> 3 -> 2 -> 1 -> 5
dummy -> 5 -> 4 -> 3 -> 2 -> 1
```

那么实现的代码就是：

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        cur = head
        while cur and cur.next:
            tmp = cur.next
            cur.next = tmp.next
            tmp.next = dummy.next
            dummy.next = tmp
        return dummy.next
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个链表。
- 空间复杂度：O(1)，原因是我们只需要使用一个 dummy node。

### [92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

这道题的描述是反转链表的一部分，与上面题目不同的是，这道题的反转是从 left 到 right，我们可以使用上面的方法来解决。

test cases:

```text
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]

Input: head = [5], left = 1, right = 1
Output: [5]
```

这道题的做法是，我们使用一个 dummy node，不过这次我们需要先找到 left 和 right 的位置，然后再进行反转。

我们可以简单用一个例子来说明这个过程：

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
dummy -> 1 -> 3 -> 2 -> 4 -> 5
dummy -> 1 -> 4 -> 3 -> 2 -> 5
```

那么实现的代码就是：

```python
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        for _ in range(left - 1):
            pre = pre.next
        cur = pre.next
        for _ in range(right - left):
            tmp = cur.next
            cur.next = tmp.next
            tmp.next = pre.next
            pre.next = tmp
        return dummy.next
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个链表。
- 空间复杂度：O(1)，原因是我们只需要使用一个 dummy node。

### [25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)

这道题的描述是反转链表的一部分，与上面题目不同的是，这道题的反转是每 k 个元素进行一次反转，我们可以使用上面的方法来解决。

test cases:

```text
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]

Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]
```

这道题的做法是，我们使用一个 dummy node，不过这次我们要先找到每 k 个元素的位置，然后再进行反转。

那么实现的代码就是：

```python
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        while True:
            cur = pre
            for _ in range(k):
                cur = cur.next
                if not cur:
                    return dummy.next
            tmp = cur.next
            cur.next = None
            cur = pre.next
            for _ in range(k - 1):
                t = cur.next
                cur.next = t.next
                t.next = pre.next
                pre.next = t
            pre = cur = tmp
        return dummy.next
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个链表。
- 空间复杂度：O(1)，原因是我们只需要使用一个 dummy node。

### [61. Rotate List](https://leetcode.com/problems/rotate-list/)

这道题的描述是旋转链表，我们可以使用上面的方法来解决。

test cases:

```text
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]

Input: head = [0,1,2], k = 4
Output: [2,0,1]
```

这道题的做法是，我们使用一个 dummy node，不过这次我们要先找到倒数第 k 个元素的位置，然后再进行反转。

那么实现的代码就是：

```python
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return head
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        cur = head
        n = 0
        while cur:
            n += 1
            cur = cur.next
        k %= n
        if k == 0:
            return head
        cur = head
        for _ in range(n - k):
            cur = cur.next
            pre = pre.next
        pre.next = None
        cur = pre.next = cur
        while cur.next:
            cur = cur.next
        cur.next = dummy.next
        return dummy.next
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个链表。
- 空间复杂度：O(1)，原因是我们只需要使用一个 dummy node。

### [143. Reorder List](https://leetcode.com/problems/reorder-list/)

这道题的描述是重排链表，我们可以使用上面的方法来解决。

test cases:

```text
Input: head = [1,2,3,4]
Output: [1,4,2,3]

Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
```

这道题的做法是，我们使用一个 dummy node，不过这次我们要先找到链表的中间位置，然后再进行反转。

那么实现的代码就是：

```python
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head:
            return head
        dummy = ListNode(0)
        dummy.next = head
        pre = cur = head
        while cur and cur.next:
            pre = pre.next
            cur = cur.next.next
        cur = pre.next
        pre.next = None
        pre = None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        cur = pre
        pre = dummy
        while head and cur:
            tmp = head.next
            head.next = cur
            cur = cur.next
            head.next.next = tmp
            head = tmp
        return dummy.next
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个链表。
- 空间复杂度：O(1)，原因是我们只需要使用一个 dummy node。

### [234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

这道题的描述是回文链表，我们可以使用上面的方法来解决。

test cases:

```text
Input: head = [1,2,2,1]
Output: true

Input: head = [1,2]
Output: false
```

这道题的做法是去判断链表的前半部分和后半部分是否相等，那么实现的代码就是：

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if not head:
            return True
        dummy = ListNode(0)
        dummy.next = head
        pre = cur = head
        while cur and cur.next:
            pre = pre.next
            cur = cur.next.next
        cur = pre.next
        pre.next = None
        pre = None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        cur = pre
        pre = dummy.next
        while cur:
            if cur.val != pre.val:
                return False
            cur = cur.next
            pre = pre.next
        return True
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个链表。
- 空间复杂度：O(1)，原因是我们只需要使用一个 dummy node。
