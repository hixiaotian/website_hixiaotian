## Foreword

In this topic, we focus on issues related to linked lists.

In fact, in many linked list problems, we can create a dummy node first. The advantage of this is that we do not need to do special processing on the head node, and it is more convenient to return at the end.

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
        # 实现过程
        # ...
        return dummy.next
```

## Basic topic

### Reverse linked list

#### [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

The description of this problem is to reverse a linked list, which we can use the above method to solve.

test cases:

```text
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Input: head = [1,2]
Output: [2,1]

Input: head = []
Output: []
```

The way to do this problem is that we use a dummy node, and then repeatedly point the cur. Next to the dummy. Next, and then point the dummy. Next to the cur. Next, so that we can complete the inversion.

We can simply use an example to illustrate this process:

```text
dummy -> 1
dummy -> 2 -> 1
dummy -> 3 -> 2 -> 1
dummy -> 4 -> 3 -> 2 -> 1
dummy -> 5 -> 4 -> 3 -> 2 -> 1
```

Then the implemented code is:

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(0)
        cur = head
        while cur:
            nxt = cur.next
            cur.next = dummy_head.next # 把当前节点的下一个节点指向dummy构建好的后面
            dummy_head.next = cur # 把dummy的下一个节点指向当前节点
            cur = nxt # 当前节点指向下一个节点
        return dummy_head.next
```

Instead of using a dummy node, Prev and nxt are used to maintain it directly.

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        cur = head
        while cur:
            nxt = cur.next
            cur.next = prev # 把当前节点的下一个节点指向prev
            prev = cur # prev指向当前节点
            cur = nxt
        return prev
```

There is also a recursive approach:

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next: # 在只剩下两个的时候，直接停止
            return head
        p = self.reverseList(head.next)
        head.next.next = head # 把当前的下下个指向自己
        head.next = None # 把当前的下一个指向None
        return p # 返回最后一个节点
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1), because we only need to use one dummy node.

#### [92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

The description of this problem is part of the reversal list. Unlike the above problem, the reversal of this problem is from left to right. We can use the above method to solve it.

test cases:

```text
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]

Input: head = [5], left = 1, right = 1
Output: [5]
```

In this problem, we use a dummy node, but this time we need to find the position of left and right first, and then reverse it.

We can simply use an example to illustrate this process:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
dummy -> 1 -> 3 -> 2 -> 4 -> 5
dummy -> 1 -> 4 -> 3 -> 2 -> 5
```

There is a good picture to explain: ![ reverse-linked-list-ii ](https://leetcode.com/uploads/files/1490008792563-reversed_linked_list.jpeg)

Then the implemented code is:

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

There is also a recursive approach (actually a backtracking approach, which is to find the position of left and right first, and then reverse it):

```python
class Solution:
    def reverseBetween(self, head, m, n):
        if not head:
            return None
        left, right = head, head
        stop = False
        def recurseAndReverse(right, m, n):
            nonlocal left, stop
            if n == 1:
                return
            # Keep moving the right pointer one step forward until (n == 1)
            right = right.next
            if m > 1:
                left = left.next
            recurseAndReverse(right, m - 1, n - 1)
            if left == right or right.next == left:
                stop = True

            # Until the boolean stop is false, swap data between the two pointers
            if not stop:
                left.val, right.val = right.val, left.val
                left = left.next
        recurseAndReverse(right, m, n)
        return head
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1), because we only need to use one dummy node.

#### [25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)

The description of this problem is part of the reversal list. Unlike the above problem, the reversal of this problem is a reversal of every K elements. We can use the above method to solve it.

test cases:

```text
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]

Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]
```

The way we do this is, we use a dummy node, but this time we're going to find the position of every K element, and then we're going to reverse it.

Then the implemented code is:

```python
class Solution(object):
    def reverseKGroup(self, head, k):
        count, node = 0, head
        while node and count < k:
            node = node.next
            count += 1
        if count < k: return head
        new_head, prev = self.reverse(head, count)
        head.next = self.reverseKGroup(new_head, k)
        return prev

    def reverse(self, head, count):
        prev, cur, nxt = None, head, head
        while count > 0:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
            count -= 1
        return (cur, prev)
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1), because we only need to use one dummy node.

### Double Pointer

#### [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

The description of this problem is to determine whether the linked list has a ring.

test cases:

```text
Input: head = [3,2,0,-4], pos = 1
Output: true

Input: head = [1,2], pos = 0
Output: true
```

In this problem, we use two pointers, a fast pointer and a slow pointer. If the fast pointer and the slow pointer meet, then there is a ring.

Then the implemented code is:

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False

        fast = slow = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

            if fast == slow:
                return True

        return False
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1). The reason is that we only need to use two pointers.

#### [876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)

The description of this problem is to find the middle node of the linked list.

test cases:

```text
Input: head = [1,2,3,4,5]
Output: [3,4,5]

Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
```

What we do in this problem is we use two pointers, a fast pointer and a slow pointer, and if the fast pointer reaches the end of the list, then the slow pointer reaches the middle of the list.

Then the implemented code is:

```python
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None

        fast = slow = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        return slow
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1). The reason is that we only need to use two pointers.

#### [160. Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)

The description of this problem is to find the intersection of two linked lists.

test cases:

```text
Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
Output: Intersected at '8'

Input: intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
Output: Intersected at '2'

Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
Output: No intersection
```

In this problem, we use two pointers, one pointer points to the linked list A, one pointer points to the linked list B, and when the pointer reaches the end of the linked list, it points to the head of the other linked list, so that when the two pointers meet, it is the intersection of the two linked lists.

Then the implemented code is:

```python
class Solution:
    def getIntersectionNode(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        if not headA or not headB:
            return None

        a, b = headA, headB

        while a != b:
            a = a.next if a else headB
            b = b.next if b else headA

        return a
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1). The reason is that we only need to use two pointers.

#### [203. Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/)

The description of this problem is to delete all nodes in the linked list that are equal to the given value Val.

test cases:

```text
Input: head = [1,2,6,3,4,5,6], val = 6
Output: [1,2,3,4,5]

Input: head = [], val = 1
Output: []

Input: head = [7,7,7,7], val = 7
Output: []
```

The solution to this problem is that we use two pointers, one pointer points to the current node, and the other pointer points to the previous node of the current node. When the value of the current node is equal to Val, we delete the current nodes, and then point the next of the previous node to the next of the present node.

Then the implemented code is:

```python
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy_head = ListNode(-1)
        dummy_head.next = head

        prev, cur = dummy_head, head

        while cur:
            if cur.val == val:
                prev.next = cur.next # 删除当前节点
            else:
                prev = prev.next # 前一个节点指向当前节点
            cur = cur.next

        return dummy_head.next
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1). The reason is that we only need to use two pointers.

#### [19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

The description of this problem is to delete the nth node from the bottom of the linked list.

test cases:

```text
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Input: head = [1], n = 1
Output: []

Input: head = [1,2], n = 1
Output: [1]
```

In this problem, we use two pointers, a fast pointer and a slow pointer. The fast pointer goes n steps first, and then the fast and slow pointers go together. When the fast pointer reaches the end of the linked list, the slow pointer reaches the nth node from the bottom of the linked list.

Then the implemented code is:

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        if not head:
            return None

        dummy = ListNode(0)
        dummy.next = head

        fast = slow = dummy

        for _ in range(n):
            fast = fast.next

        while fast and fast.next:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next

        return dummy.next
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1). The reason is that we only need to use two pointers.

#### [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

The description of this problem is to merge two ordered linked lists.

test cases:

```text
Input: l1 = [1,2,4], l2 = [1,3,4]
Output: [1,1,2,3,4,4]

Input: l1 = [], l2 = []
Output: []

Input: l1 = [], l2 = [0]
Output: [0]
```

In this problem, we use a dummy node, then traverse two linked lists, add the values of the two linked lists, and then add the result to the dummy node.

Then the implemented code is:

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(-1)
        prev = dummy_head

        while list1 and list2:
            if list1.val <= list2.val:
                prev.next = list1
                list1 = list1.next
                prev = prev.next
            else:
                prev.next = list2
                list2 = list2.next
                prev = prev.next

        if list1 and not list2:
            prev.next = list1
            list1 = list1.next
        if list2 and not list1:
            prev.next = list2
            list2 = list2.next

        return dummy_head.next
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1), because we only need to use one dummy node.

#### [2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)

The description of this problem is that given two linked lists, each of which represents a number, we need to add the two numbers and return a new linked list.

test cases:

```text
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]

Input: l1 = [0], l2 = [0]
Output: [0]

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
```

In this problem, we use a dummy node, then traverse two linked lists, add the values of the two linked lists, and then add the result to the dummy node.

Then the implemented code is:

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(0)
        cur = dummy_head
        carry = 0

        while l1 or l2 or carry:
            l1_val = l1.val if l1 else 0
            l2_val = l2.val if l2 else 0
            cur_sum = l1_val + l2_val + carry
            carry = cur_sum // 10
            new_node = ListNode(cur_sum % 10)

            cur.next = new_node
            cur = new_node
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return dummy_head.next
```

#### [328. Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/)

The description of this problem is that given a linked list, we need to put the odd nodes of the linked list in front of the even nodes.

test cases:

```text
Input: head = [1,2,3,4,5]
Output: [1,3,5,2,4]

Input: head = [2,1,3,5,6,4,7]
Output: [2,3,6,7,1,5,4]
```

In this problem, we use two dummy nodes, then traverse the linked list, put the odd nodes after the first dummy node, and put the even nodes after the second dummy node. Finally, connect the tail of the first dummy node to the head of the second dummy node.

Then the implemented code is:

```python
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None

        dummy = ListNode(-1)
        dummy.next = head

        odd, even = head, head.next
        connection_point = even

        while even and even.next:
            odd.next = odd.next.next
            odd = odd.next

            even.next = even.next.next
            even = even.next

        odd.next = connection_point

        return dummy.next
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1), because we only need to use two dummy nodes.

#### [86. Partition List](https://leetcode.com/problems/partition-list/)

The description of this problem is that given a linked list and a value X, we need to put the nodes less than X in the linked list in front of the nodes greater than or equal to X.

test cases:

```text
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]

Input: head = [2,1], x = 2
Output: [1,2]
```

In this problem, we use two dummy nodes, then traverse the linked list, put the nodes less than X behind the first dummy node, and put the nodes greater than or equal to X behind the second dummy node. Finally, connect the tail of the first dummy node to the head of the second dummy node.

Then the implemented code is:

```python
def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
    if not head:
        return None

    head1 = cur1 = ListNode()
    head2 = cur2 = ListNode()

    while head:
        if head.val < x:
            cur1.next = head
            cur1 = cur1.next
        else:
            cur2.next = head
            cur2 = cur2.next
        head = head.next

    cur1.next = head2.next
    cur2.next = None

    return head1.next
```

### Other types

#### [82. Remove Duplicates from Sorted List II](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/)

The description of this problem is to delete the duplicate elements in the linked list.

test cases:

```text
Input: head = [1,2,3,3,4,4,5]
Output: [1,2,5]

Input: head = [1,1,1,2,3]
Output: [2,3]
```

To solve this problem, we use a dummy node and then traverse the linked list. If the value of the current node is the same as that of the next node, then we need to delete the two nodes. If they are not the same, then we add the current node to the dummy node.

Then the implemented code is:

```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None

        dummy = ListNode(0)
        dummy.next = head

        pre = dummy
        cur = head

        while cur:
            while cur.next and cur.val == cur.next.val:
                cur = cur.next

            if pre.next == cur:
                pre = pre.next
            else:
                pre.next = cur.next

            cur = cur.next

        return dummy.next
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1). The reason is that we only need to use two pointers.

#### [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

The description of this problem is to exchange two adjacent elements in the linked list.

test cases:

```text
Input: head = [1,2,3,4]
Output: [2,1,4,3]

Input: head = []
Output: []

Input: head = [1]
Output: [1]
```

The way we do this is, we use a dummy node, but this time we're going to find the position of every two elements, and then we're going to reverse it.

Then the implemented code is:

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        while pre.next and pre.next.next:
            first = pre.next
            second = pre.next.next

            # 把三个node（pre，first，second）重新用他们的next链接，注意顺序!
            pre.next = second
            first.next = second.next
            second.next = first

            # pre走到下一个
            pre = first
        return dummy.next

```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1), because we only need to use one dummy node.

## Advanced combination questions

#### [61. Rotate List](https://leetcode.com/problems/rotate-list/)

The description of this problem is to rotate the list, and we can use the above method to solve it.

test cases:

```text
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]

Input: head = [0,1,2], k = 4
Output: [2,0,1]
```

In this problem, we use a dummy node to actively find the length of the linked list to form a ring, then find the previous node of the new head node, and then break the ring.

Then the implemented code is:

```python
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        cur = head
        length = 1 # 自己也算一个

        # 拿到链表的长度
        while cur.next:
            cur = cur.next
            length += 1

        # 把链表连成环
        cur.next = head

        # 求余拿到真正的k
        k = k % length

        # 找到新的头结点的前一个节点
        for i in range(length - k):
            cur = cur.next

        head = cur.next
        cur.next = None # 断开环
        return head
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1), because we only need to use one dummy node.

#### [143. Reorder List](https://leetcode.com/problems/reorder-list/)

The description of this problem is to rearrange the linked list, and we can use the above method to solve it.

test cases:

```text
Input: head = [1,2,3,4]
Output: [1,4,2,3]

Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
```

To solve this problem, we use a dummy node and then divide it into three steps:

1. Find the middle of the list
2. Reverse the second half
3. Reconnect

To use an example to explain:

```text
1 -> 2 -> 3 -> 4 -> 5

1. 找到链表的中间位置
slow = 3

2. 反转后半部分
1 -> 2    5 -> 4 -> 3

3. 重新连接
1 -> 5 -> 2 -> 4 -> 3
```

Then the implemented code is:

```python
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head:
            return None

        # 找到链表的中间位置
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # 反转后半部分
        prev = None
        cur = slow
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        # 重新连接
        first, second = head, prev
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1), because we only need to use one dummy node.

#### [234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

The description of this problem is a palindrome list, and we can use the above method to solve it.

test cases:

```text
Input: head = [1,2,2,1]
Output: true

Input: head = [1,2]
Output: false
```

To solve this problem, we use a dummy node and then divide it into three steps:

1. Find the middle of the list
2. Reverse the second half
3. Determine whether they are equal

```python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:

        # 找到链表的中间位置
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        # 如果是奇数，那么slow再往前走一步，要不然会多一个
        if fast:
            slow = slow.next

        # 反转后半部分
        def reverse(head):
            if not head:
                return None

            prev = None
            cur = head
            while cur:
                nxt = cur.next
                cur.next = prev
                prev = cur
                cur = nxt
            return prev

        # 判断是否相等
        new_head = reverse(slow)
        while new_head:
            if head.val != new_head.val:
                return False
            head = head.next
            new_head = new_head.next
        return True
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire linked list.
- Space complexity: O (1), because we only need to use one dummy node.

```

```
