### 前言

LeetCode 中有很多类似 OOD 的设计的题目，有些面试官非常喜欢考察此类题目，因为这类题目可以考察面试者的设计能力同时还能验证算法能力，这里我们列举几个比较重要的设计题，供大家参考。

### 基础题

#### [146. LRU Cache](https://leetcode.com/problems/lru-cache/)

这道题的描述是，设计一个 LRU Cache，这个 Cache 支持两个操作，一个是 put，一个是 get，其中 put 操作是插入键值对，如果键值对的 key 已经存在，那么就更新 value，如果不存在，那么就插入这个键值对，如果插入之后，Cache 的大小超过了 Cache 的容量，那么就删除最近最少使用的键值对，get 操作是获取键值对，如果键值对存在，那么就返回 value，否则就返回 -1。

test cases:

```text
Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
```

这道题的思路是，我们使用一个双向链表来存储键值对，链表的头部是最近最少使用的键值对，链表的尾部是最近最多使用的键值对，同时我们使用一个哈希表来存储键值对，这样我们就可以在 O(1) 的时间内找到键值对，这里我们使用一个哈希表来存储键值对的原因是，我们需要在 O(1) 的时间内找到键值对，而链表的查找时间复杂度是 O(n)，所以我们使用哈希表来存储键值对，这样我们就可以在 O(1) 的时间内找到键值对。

```python
class Node:
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.dummy_left = Node()
        self.dummy_right = Node()
        self.dummy_left.next = self.dummy_right
        self.dummy_left.prev = self.dummy_right
        self.dummy_right.next = self.dummy_left
        self.dummy_right.prev = self.dummy_left

    def add_element(self, node: Node):
        last_element = self.dummy_right.prev
        self.dummy_right.prev = node
        node.next = self.dummy_right
        last_element.next = node
        node.prev = last_element

    def remove_element(self, node: Node):
        node.next.prev = node.prev
        node.prev.next = node.next

    def pop_element(self):
        first_element = self.dummy_left.next
        self.remove_element(first_element)
        return first_element.key

    def put_to_left(self, node):
        self.remove_element(node)
        self.add_element(node)

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self.put_to_left(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            node = Node(key, value)
            if len(self.cache) == self.capacity:
                remove_key = self.pop_element()
                del self.cache[remove_key]
            self.add_element(node)
            self.cache[key] = node
        else:
            self.cache[key].value = value
            self.put_to_left(self.cache[key])

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

#### [705. Design HashSet](https://leetcode.com/problems/design-hashset/)

这道题的描述是，设计一个哈希集合，这个哈希集合支持 add，remove，contains 操作，其中 add 是插入元素，remove 是删除元素，contains 是判断元素是否存在。

test cases:

```text
Input
["MyHashSet", "add", "add", "contains", "contains", "add", "contains", "remove", "contains"]
[[], [1], [2], [1], [3], [2], [2], [2], [2]]
Output
[null, null, null, true, false, null, true, null, false]

Explanation
MyHashSet myHashSet = new MyHashSet();
myHashSet.add(1);      // set = [1]
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(1); // return True
myHashSet.contains(3); // return False, (not found)
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(2); // return True
myHashSet.remove(2);   // set = [1]
myHashSet.contains(2); // return False, (already removed)
```

这道题的思路是，我们使用一个数组来存储元素，数组的下标是元素的值，数组的值是元素是否存在，这样我们就可以在 O(1) 的时间内判断元素是否存在，同时我们使用一个哈希函数来计算元素的下标，这样我们就可以在 O(1) 的时间内插入元素和删除元素。

```python
class MyHashSet:
    def __init__(self):
        self.hash_set = [False] * 1000001

    def add(self, key: int) -> None:
        self.hash_set[key] = True

    def remove(self, key: int) -> None:
        self.hash_set[key] = False

    def contains(self, key: int) -> bool:
        return self.hash_set[key]
```

但是这样做尽管能过 test cases，却不符合 hashset 的实现逻辑，并且一个 key 会对应一个 bucket，而不是只有一个元素，肯定会造成冲突的情况。那么我们可以把一个 bucket 里设计成一个链表，这样就可以解决冲突的问题了。

```python
class Node:
    def __init__(self, value, nextNode=None):
        self.value = value
        self.next = nextNode

class Bucket:
    def __init__(self):
        # a pseudo head
        self.head = Node(0)

    def insert(self, newValue):
        # if not existed, add the new element to the head.
        if not self.exists(newValue):
            newNode = Node(newValue, self.head.next)
            # set the new head.
            self.head.next = newNode

    def delete(self, value):
        prev = self.head
        curr = self.head.next
        while curr is not None:
            if curr.value == value:
                # remove the current node
                prev.next = curr.next
                return
            prev = curr
            curr = curr.next

    def exists(self, value):
        curr = self.head.next
        while curr is not None:
            if curr.value == value:
                # value existed already, do nothing
                return True
            curr = curr.next
        return False

class MyHashSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.keyRange = 769
        self.bucketArray = [Bucket() for i in range(self.keyRange)]

    def _hash(self, key):
        return key % self.keyRange

    def add(self, key):
        """
        :type key: int
        :rtype: None
        """
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].insert(key)

    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].delete(key)

    def contains(self, key):
        """
        Returns true if this set contains the specified element
        :type key: int
        :rtype: bool
        """
        bucketIndex = self._hash(key)
        return self.bucketArray[bucketIndex].exists(key)


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
```

其中一个优化是我们可以通过二分法来在每一个 bucket 里进行查找，那么每一个 bucket 当成 BST 来处理，这样就可以把时间复杂度降低到 O(logN)。

```python
class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

class BSTree:
    def __init__(self):
        self.root = None

    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if root is None or val == root.val:
            return root

        return self.searchBST(root.left, val) if val < root.val \
            else self.searchBST(root.right, val)

    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)

        if val > root.val:
            # insert into the right subtree
            root.right = self.insertIntoBST(root.right, val)
        elif val == root.val:
            return root
        else:
            # insert into the left subtree
            root.left = self.insertIntoBST(root.left, val)
        return root

    def successor(self, root):
        """
        One step right and then always left
        """
        root = root.right
        while root.left:
            root = root.left
        return root.val

    def predecessor(self, root):
        """
        One step left and then always right
        """
        root = root.left
        while root.right:
            root = root.right
        return root.val

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return None

        # delete from the right subtree
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        # delete from the left subtree
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        # delete the current node
        else:
            # the node is a leaf
            if not (root.left or root.right):
                root = None
            # the node is not a leaf and has a right child
            elif root.right:
                root.val = self.successor(root)
                root.right = self.deleteNode(root.right, root.val)
            # the node is not a leaf, has no right child, and has a left child
            else:
                root.val = self.predecessor(root)
                root.left = self.deleteNode(root.left, root.val)

        return root

class MyHashSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.keyRange = 769
        self.bucketArray = [Bucket() for i in range(self.keyRange)]

    def _hash(self, key) -> int:
        return key % self.keyRange

    def add(self, key: int) -> None:
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].insert(key)

    def remove(self, key: int) -> None:
        """
        :type key: int
        :rtype: None
        """
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].delete(key)

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        :type key: int
        :rtype: bool
        """
        bucketIndex = self._hash(key)
        return self.bucketArray[bucketIndex].exists(key)

class Bucket:
    def __init__(self):
        self.tree = BSTree()

    def insert(self, value):
        self.tree.root = self.tree.insertIntoBST(self.tree.root, value)

    def delete(self, value):
        self.tree.root = self.tree.deleteNode(self.tree.root, value)

    def exists(self, value):
        return (self.tree.searchBST(self.tree.root, value) is not None)

# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
```

#### [706. Design HashMap](https://leetcode.com/problems/design-hashmap/)

这道题的描述是，设计一个哈希映射，这个哈希映射支持 put，get，remove 操作，其中 put 是插入键值对，get 是获取键值对，remove 是删除键值对。

test cases:

```text
Input
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
Output
[null, null, null, 1, -1, null, 1, null, -1]

Explanation
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // The map is now [[1,1]]
myHashMap.put(2, 2); // The map is now [[1,1], [2,2]]
myHashMap.get(1);    // return 1, The map is now [[1,1], [2,2]]
myHashMap.get(3);    // return -1 (i.e., not found), The map is now [[1,1], [2,2]]
myHashMap.put(2, 1); // The map is now [[1,1], [2,1]] (i.e., update the existing value)
myHashMap.get(2);    // return 1, The map is now [[1,1], [2,1]]
myHashMap.remove(2); // remove the mapping for 2, The map is now [[1,1]]
myHashMap.get(2);    // return -1 (i.e., not found), The map is now [[1,1]]
```

这道题的思路是，我们使用一个数组来存储键值对，数组的下标是键的哈希值，数组的值是键值对，这样我们就可以在 O(1) 的时间内插入键值对和删除键值对，同时我们使用一个哈希函数来计算键的哈希值，这样我们就可以在 O(1) 的时间内获取键值对。

```python
class Bucket:
    def __init__(self):
        self.bucket = []

    def get(self, key):
        for (k, v) in self.bucket:
            if k == key:
                return v
        return -1

    def update(self, key, value):
        found = False
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                self.bucket[i] = (key, value)
                found = True
                break
        if not found:
            self.bucket.append((key, value))

    def remove(self, key):
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                del self.bucket[i]

class MyHashMap:

    def __init__(self):
        self.key_space = 2069
        self.hash_table = [Bucket() for i in range(self.key_space)]

    def put(self, key: int, value: int) -> None:
        hash_key = key % self.key_space
        self.hash_table[hash_key].update(key, value)

    def get(self, key: int) -> int:
        hash_key = key % self.key_space
        return self.hash_table[hash_key].get(key)

    def remove(self, key: int) -> None:
        hash_key = key % self.key_space
        self.hash_table[hash_key].remove(key)



# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
```

当然这个 bucket 也可以像前面一样，用链表来实现，时间复杂度不变。

```python
class Node:
    def __init__(self, key, val, next=None):
        self.key = key
        self.val = val
        self.next = next

class Bucket:
    def __init__(self):
        # a pseudo head
        self.head = Node(-1, -1)

    def insert(self, key, val):
        # if not existed, add the new element to the head.
        if not self.exists(key):
            new_node = Node(key, val, self.head.next)
            # set the new head.
            self.head.next = new_node

    def update(self, key, val):
        cur = self.head.next
        while cur:
            if cur.key == key:
                cur.val = val
                return
            cur = cur.next
        self.insert(key, val)
        return True

    def delete(self, key):
        prev = self.head
        curr = self.head.next
        while curr:
            if curr.key == key:
                prev.next = curr.next
                return
            prev = prev.next
            curr = curr.next

    def exists(self, key):
        curr = self.head.next
        while curr is not None:
            if curr.key == key:
                # value existed already, do nothing
                return True
            curr = curr.next
        return False

    def get(self, key):
        cur = self.head.next
        while cur:
            if cur.key == key:
                return cur.val
            cur = cur.next
        return -1

class MyHashMap:

    def __init__(self):
        self.key_space = 2069
        self.hash_table = [Bucket() for i in range(self.key_space)]

    def put(self, key: int, value: int) -> None:
        hash_key = key % self.key_space
        self.hash_table[hash_key].update(key, value)

    def get(self, key: int) -> int:
        hash_key = key % self.key_space
        return self.hash_table[hash_key].get(key)

    def remove(self, key: int) -> None:
        hash_key = key % self.key_space
        self.hash_table[hash_key].delete(key)



# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
```

#### [155. Min Stack](https://leetcode.com/problems/min-stack/)

这道题的描述是，设计一个栈，这个栈支持 push，pop，top，getMin 操作，其中 push 是插入元素，pop 是删除栈顶元素，top 是获取栈顶元素，getMin 是获取栈中最小元素。

test cases:

```text
Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
```

这道题的思路是，我们使用两个栈，一个栈用来存储元素，另一个栈用来存储最小元素，这样我们就可以在 O(1) 的时间内获取最小元素。

```python
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
            return

        current_min = self.stack[-1][1]
        self.stack.append((val, min(val, current_min)))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

#### [232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)

这道题的描述是，设计一个队列，这个队列支持 push，pop，peek，empty 操作，其中 push 是插入元素，pop 是删除队首元素，peek 是获取队首元素，empty 是判断队列是否为空。

test cases:

```text
Input
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]

Output
[null, null, null, 1, 1, false]

Explanation
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
```

这道题的思路是，我们使用两个栈，一个栈用来存储元素，另一个栈用来存储元素的逆序，这样我们就可以在 O(1) 的时间内获取队首元素。

```python
class MyQueue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def push(self, x):
        # 举一个例子，假设 s1 = [3, 2, 1]，s2 = []，现在要 push 4
        # 那么我们先把 s1 中的元素全部 pop 出来，再 push 到 s2 中，这样 s2 = [1, 2, 3]
        # 然后我们再 push 4 到 s1 中，这样 s1 = [4, 3, 2, 1]
        while self.s1:
            self.s2.append(self.s1.pop())
        self.s1.append(x)
        while self.s2:
            self.s1.append(self.s2.pop())

    def pop(self):
        return self.s1.pop()

    def peek(self):
        return self.s1[-1]

    def empty(self):
        return not self.s1



# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```

#### [225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)

这道题的描述是，设计一个栈，这个栈支持 push，pop，top，empty 操作，其中 push 是插入元素，pop 是删除栈顶元素，top 是获取栈顶元素，empty 是判断栈是否为空。

test cases:

```text
Input
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 2, 2, false]

Explanation
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // return 2
myStack.pop(); // return 2
myStack.empty(); // return False
```

这道题的思路是，我们使用两个队列，一个队列用来存储元素，另一个队列用来存储元素的逆序，这样我们就可以在 O(1) 的时间内获取栈顶元素。

```python
class MyStack:
    def __init__(self):
        self.q1 = []
        self.q2 = []

    def push(self, x):
        self.q1.append(x)

    def pop(self):
        # 举一个例子，假设 q1 = [1, 2, 3]，q2 = []，现在要 pop
        # 那么我们先把 q1 中的元素全部 pop 出来，再 push 到 q2 中，这样 q2 = [1, 2, 3]
        # 然后我们再 pop q2，这样 q2 = []，q1 = [1, 2]
        while len(self.q1) > 1:
            self.q2.append(self.q1.pop(0))

        # 交换 q1 和 q2，这样 q1 = []，q2 = [1, 2]
        self.q1, self.q2 = self.q2, self.q1
        return self.q2.pop()

    def top(self):
        return self.q1[-1]

    def empty(self):
        return not self.q1
```

当然也可以直接使用一个队列，每次 push 的时候，把队列的元素逆序一下，这样队列的第一个元素就是栈顶元素了。

```python
class Stack:
    def __init__(self):
        self._queue = collections.deque()

    def push(self, x):
        # 举一个例子，假设 q = [1, 2, 3]，现在要 push 4
        # 那么我们先把 4 push 到 q 中，这样 q = [1, 2, 3, 4]
        # 然后我们把 q 中的元素全部 pop 出来，再 push 到 q 中，这样 q = [4, 1, 2, 3]
        q = self._queue
        q.append(x)
        for _ in range(len(q) - 1):
            q.append(q.popleft())

    def pop(self):
        return self._queue.popleft()

    def top(self):
        return self._queue[0]

    def empty(self):
        return not len(self._queue)
```

#### [622. Design Circular Queue](https://leetcode.com/problems/design-circular-queue/)

这道题的描述是，设计一个循环队列，这个循环队列支持 enQueue，deQueue，Front，Rear，isEmpty，isFull 操作，其中 enQueue 是插入元素，deQueue 是删除队首元素，Front 是获取队首元素，Rear 是获取队尾元素，isEmpty 是判断队列是否为空，isFull 是判断队列是否已满。

test cases:

```text
Input
["MyCircularQueue", "enQueue", "enQueue", "enQueue", "enQueue", "Rear", "isFull", "deQueue", "enQueue", "Rear"]
[[3], [1], [2], [3], [4], [], [], [], [4], []]
Output
[null, true, true, true, false, 3, true, true, true, 4]

Explanation
MyCircularQueue myCircularQueue = new MyCircularQueue(3);
myCircularQueue.enQueue(1); // return True
myCircularQueue.enQueue(2); // return True
myCircularQueue.enQueue(3); // return True
myCircularQueue.enQueue(4); // return False
myCircularQueue.Rear();     // return 3
myCircularQueue.isFull();   // return True
myCircularQueue.deQueue();  // return True
myCircularQueue.enQueue(4); // return True
myCircularQueue.Rear();     // return 4
```

这道题的思路是，我们使用一个数组来存储元素，使用两个指针来分别指向队首和队尾，这样我们就可以在 O(1) 的时间内完成所有操作。

```python
from threading import Lock
class MyCircularQueue:

    def __init__(self, k: int):
        self.queue = [0] * k
        self.head = 0
        self.count = 0
        self.capacity = k
        self.queueLock = Lock()

    def enQueue(self, value: int) -> bool:
        with self.queueLock:
            if self.count == self.capacity:
                return False

            self.queue[(self.head + self.count) % self.capacity] = value
            self.count += 1
        return True

    def deQueue(self) -> bool:
        with self.queueLock:
            if self.isEmpty():
                return False
            self.head = (self.head + 1) % self.capacity
            self.count -= 1
        return True

    def Front(self) -> int:
        return -1 if self.isEmpty() else self.queue[self.head]

    def Rear(self) -> int:
        return -1 if self.isEmpty() else self.queue[(self.head + self.count - 1) % self.capacity]

    def isEmpty(self) -> bool:
        return self.count == 0

    def isFull(self) -> bool:
        return not self.isEmpty() and self.count == self.capacity



# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
```

#### [641. Design Circular Deque](https://leetcode.com/problems/design-circular-deque/)

这道题的描述是，设计一个循环双端队列，这个循环双端队列支持 insertFront，insertLast，deleteFront，deleteLast，getFront，getRear，isEmpty，isFull 操作，其中 insertFront 是插入队首元素，insertLast 是插入队尾元素，deleteFront 是删除队首元素，deleteLast 是删除队尾元素，getFront 是获取队首元素，getRear 是获取队尾元素，isEmpty 是判断队列是否为空，isFull 是判断队列是否已满。

test cases:

```text
Input
["MyCircularDeque", "insertLast", "insertLast", "insertFront", "insertFront", "getRear", "isFull", "deleteLast", "insertFront", "getFront"]
[[3], [1], [2], [3], [4], [], [], [], [4], []]
Output
[null, true, true, true, false, 2, true, true, true, 4]

Explanation
MyCircularDeque myCircularDeque = new MyCircularDeque(3);
myCircularDeque.insertLast(1);  // return True
myCircularDeque.insertLast(2);  // return True
myCircularDeque.insertFront(3); // return True
myCircularDeque.insertFront(4); // return False, the queue is full.
myCircularDeque.getRear();      // return 2
myCircularDeque.isFull();       // return True
myCircularDeque.deleteLast();   // return True
myCircularDeque.insertFront(4); // return True
myCircularDeque.getFront();     // return 4
```

这道题的思路是，我们使用一个数组来存储元素，使用两个指针来分别指向队首和队尾，这样我们就可以在 O(1) 的时间内完成所有操作。

```python
class MyCircularDeque:

    def __init__(self, k: int):
        self.queue = [None] * k
        self.size = 0
        self.front = 0
        self.capacity = k

    def debug(self):
        print(self.queue)
        print(self.size)
        print(self.front)

    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        if self.isEmpty():
            self.queue[self.front] = value
        else:
            self.front = (self.front + self.capacity - 1) % self.capacity
            self.queue[self.front] = value
        self.size += 1
        # self.debug()
        return True

    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        if self.isEmpty():
            self.queue[self.front] = value
        else:
            self.queue[(self.front + self.size) % self.capacity] = value
        self.size += 1
        # self.debug()
        return True

    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        self.size -= 1
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.debug()
        return True

    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        self.queue[(self.front + self.size - 1) % self.capacity] = None
        self.size -= 1
        # self.debug()
        return True

    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        # self.debug()
        return self.queue[self.front]

    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        # self.debug()
        return self.queue[(self.front + self.size - 1) % self.capacity]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == self.capacity


# Your MyCircularDeque object will be instantiated and called as such:
# obj = MyCircularDeque(k)
# param_1 = obj.insertFront(value)
# param_2 = obj.insertLast(value)
# param_3 = obj.deleteFront()
# param_4 = obj.deleteLast()
# param_5 = obj.getFront()
# param_6 = obj.getRear()
# param_7 = obj.isEmpty()
# param_8 = obj.isFull()
```

#### [1166. Design File System](https://leetcode.com/problems/design-file-system/)

这道题的描述是，设计一个文件系统，这个文件系统支持 createPath 和 get 函数，其中 createPath 是创建路径，get 是获取路径的值。

test cases:

```text
Input:
["FileSystem","createPath","createPath","get","createPath","get"]
[[],["/leet",1],["/leet/code",2],["/leet/code"],["/c/d",1],["/c"]]
Output:
[null,true,true,2,false,-1]
Explanation:
FileSystem fileSystem = new FileSystem();

fileSystem.createPath("/leet", 1); // return true
fileSystem.createPath("/leet/code", 2); // return true
fileSystem.get("/leet/code"); // return 2
fileSystem.createPath("/c/d", 1); // return false because the parent path "/c" doesn't exist.
fileSystem.get("/c"); // return -1 because this path doesn't exist.
```

这道题的思路是，我们使用一个字典来存储路径和值的映射，使用一个集合来存储已经存在的路径，这样我们就可以在 O(1) 的时间内完成所有操作。

```python
class TrieNode:
    def __init__(self, name: str = ""):
        self.children = collections.defaultdict(TrieNode)
        self.name = name
        self.value = -1

class FileSystem:

    def __init__(self):
        self.root = TrieNode()

    def createPath(self, path: str, value: int) -> bool:
        cur = self.root
        components = path.split("/")

        for i in range(1, len(components)):
            name = components[i]

            if name not in cur.children:
                if i == len(components) - 1:
                    cur.children[name] = TrieNode(name)
                else:
                    return False
            cur = cur.children[name]

        if cur.value != -1:
            return False

        cur.value = value
        return True

    def get(self, path: str) -> int:
        cur = self.root
        components = path.split("/")

        for i in range(1, len(components)):
            name = components[i]
            if name not in cur.children:
                return -1
            cur = cur.children[name]
        return cur.value



# Your FileSystem object will be instantiated and called as such:
# obj = FileSystem()
# param_1 = obj.createPath(path,value)
# param_2 = obj.get(path)
```

#### [588. Design In-Memory File System](https://leetcode.com/problems/design-in-memory-file-system/)

这道题的描述是，设计一个内存文件系统，这个文件系统支持 ls，mkdir，addContentToFile，readContentFromFile 四个操作，其中 ls 是列出当前目录下的文件和目录，mkdir 是创建目录，addContentToFile 是向文件中添加内容，readContentFromFile 是读取文件的内容。

test cases:

```text
Input
["FileSystem", "ls", "mkdir", "addContentToFile", "ls", "readContentFromFile"]
[[], ["/"], ["/a/b/c"], ["/a/b/c/d", "hello"], ["/"], ["/a/b/c/d"]]
Output
[null, [], null, null, ["a"], "hello"]

Explanation
FileSystem fileSystem = new FileSystem();
fileSystem.ls("/");                         // return []
fileSystem.mkdir("/a/b/c");
fileSystem.addContentToFile("/a/b/c/d", "hello");
fileSystem.ls("/");                         // return ["a"]
fileSystem.readContentFromFile("/a/b/c/d"); // return "hello"
```

这道题的思路是，我们使用一个字典来存储路径和值的映射，使用一个集合来存储已经存在的路径，这样我们就可以在 O(1) 的时间内完成所有操作。

```python
class TrieNode:
    def __init__(self):
        self.content = ""
        self.children = collections.defaultdict(TrieNode)
        self.isfile = False

class FileSystem:
    def __init__(self):
        self.root = TrieNode()

    def ls(self, path: str) -> List[str]:
        path_list = path.split("/")
        cur = self.root
        for path in path_list:
            if not path:
                continue
            cur = cur.children.get(path)

        if cur.isfile:
            return [path]

        ans = [i for i in cur.children.keys()]
        if not ans:
            return ans
        ans.sort()
        return ans

    def mkdir(self, path: str) -> None:
        path_list = path.split("/")
        cur = self.root
        for path in path_list:
            if not path:
                continue
            cur = cur.children[path]

    def addContentToFile(self, filePath: str, content: str) -> None:
        path_list = filePath.split("/")
        cur = self.root
        for path in path_list:
            if not path:
                continue
            cur = cur.children[path]
        cur.content += content
        cur.isfile = True

    def readContentFromFile(self, filePath: str) -> str:
        path_list = filePath.split("/")
        cur = self.root
        for path in path_list:
            if not path:
                continue
            cur = cur.children.get(path)
        return cur.content



# Your FileSystem object will be instantiated and called as such:
# obj = FileSystem()
# param_1 = obj.ls(path)
# obj.mkdir(path)
# obj.addContentToFile(filePath,content)
# param_4 = obj.readContentFromFile(filePath)
```

#### [1472. Design Browser History](https://leetcode.com/problems/design-browser-history/)

这道题的描述是，设计一个浏览器历史记录，这个浏览器历史记录支持 visit，back，forward 操作，其中 visit 是访问一个新的网页，back 是回退到上一个网页，forward 是前进到下一个网页。

test cases:

```text
Input:
["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
[["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
Output:
[null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]

Explanation:
BrowserHistory browserHistory = new BrowserHistory("leetcode.com");
browserHistory.visit("google.com");       // You are in "leetcode.com". Visit "google.com"
browserHistory.visit("facebook.com");     // You are in "google.com". Visit "facebook.com"
browserHistory.visit("youtube.com");      // You are in "facebook.com". Visit "youtube.com"
browserHistory.back(1);                   // You are in "youtube.com", move back to "facebook.com" return "facebook.com"
browserHistory.back(1);                   // You are in "facebook.com", move back to "google.com" return "google.com"
browserHistory.forward(1);                // You are in "google.com", move forward to "facebook.com" return "facebook.com"
browserHistory.visit("linkedin.com");     // You are in "facebook.com". Visit "linkedin.com"
browserHistory.forward(2);                // You are in "linkedin.com", you cannot move forward any steps.
browserHistory.back(2);                   // You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com". return "google.com"
browserHistory.back(7);                   // You are in "google.com", you can move back only one step to "leetcode.com". return "leetcode.com"
```

这道题的思路是，我们使用两个栈来存储历史记录，一个栈用来存储当前的历史记录，一个栈用来存储前进的历史记录，这样我们就可以在 O(1) 的时间内完成所有操作。

```python
class BrowserHistory:

    def __init__(self, homepage: str):
        self.history_stack = []
        self.future_stack = []
        self.cur = homepage

    def visit(self, url: str) -> None:
        self.history_stack.append(self.cur)
        self.cur = url
        self.future_stack = []

    def back(self, steps: int) -> str:
        while steps > 0 and self.history_stack:
            self.future_stack.append(self.cur)
            self.cur = self.history_stack.pop()
            steps -= 1
        return self.cur

    def forward(self, steps: int) -> str:
        while steps > 0 and self.future_stack:
            self.history_stack.append(self.cur)
            self.cur = self.future_stack.pop()
            steps -= 1
        return self.cur

# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)
```

#### [355. Design Twitter](https://leetcode.com/problems/design-twitter/)

这道题的描述是，设计一个 Twitter，这个 Twitter 支持 postTweet，getNewsFeed，follow，unfollow 操作，其中 postTweet 是发推，getNewsFeed 是获取最近的 10 条推文，follow 是关注某个用户，unfollow 是取消关注某个用户。

test cases:

```text
Input
["Twitter", "postTweet", "getNewsFeed", "follow", "postTweet", "getNewsFeed", "unfollow", "getNewsFeed"]
[[], [1, 5], [1], [1, 2], [2, 6], [1], [1, 2], [1]]
Output
[null, null, [5], null, null, [6, 5], null, [5]]

Explanation
Twitter twitter = new Twitter();
twitter.postTweet(1, 5); // User 1 posts a new tweet (id = 5).
twitter.getNewsFeed(1);  // User 1's news feed should return a list with 1 tweet id -> [5]. return [5]
twitter.follow(1, 2);    // User 1 follows user 2.
twitter.postTweet(2, 6); // User 2 posts a new tweet (id = 6).
twitter.getNewsFeed(1);  // User 1's news feed should return a list with 2 tweet ids -> [6, 5]. Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.
twitter.unfollow(1, 2);  // User 1 unfollows user 2.
twitter.getNewsFeed(1);  // User 1's news feed should return a list with 1 tweet id -> [5], since user 1 is no longer following user 2.
```

这道题的思路是，我们使用一个字典来存储用户和他关注的用户，使用一个列表来存储用户的推文，每个推文是一个元组，第一个元素是推文的 id，第二个元素是推文的时间戳，我们使用一个变量来记录推文的时间戳，每次发推的时候，我们就将时间戳加一，这样就可以保证推文的时间戳是递增的。

```python
class Twitter:

    def __init__(self):
        self.relation_graph = collections.defaultdict(set)
        self.post = collections.defaultdict(list)
        self.timer = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.timer -= 1
        self.post[userId].append([self.timer, tweetId])

    def getNewsFeed(self, userId: int) -> List[int]:
        feeds = []
        for followee in self.relation_graph[userId]:
            feeds += self.post[followee]
        feeds += self.post[userId]
        return [feed for _, feed in heapq.nsmallest(10, feeds)]

    def follow(self, followerId: int, followeeId: int) -> None:
        self.relation_graph[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.relation_graph[followerId]:
            self.relation_graph[followerId].remove(followeeId)

# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)
```

#### [380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/)

这道题的描述是，设计一个数据结构，这个数据结构支持 insert，remove，getRandom 操作，其中 insert 和 remove 的时间复杂度都是 O(1)，getRandom 的时间复杂度是 O(1)。

test cases:

```text
["RandomizedSet","insert","remove","insert","getRandom","remove","insert","getRandom"]
[[],[1],[2],[2],[],[1],[2],[]]
```

这道题的思路是，我们使用一个列表来存储数据，使用一个字典来存储数据和它在列表中的索引，这样我们就可以在 O(1) 的时间复杂度内完成 insert 和 remove 操作，而 getRandom 操作就是随机返回列表中的一个元素。

```python
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = []
        self.index = {}

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.index:
            return False
        self.data.append(val)
        self.index[val] = len(self.data) - 1
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.index:
            return False
        index = self.index[val]
        self.data[index] = self.data[-1]
        self.index[self.data[-1]] = index
        self.data.pop()
        self.index.pop(val)
        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return random.choice(self.data)
```

#### [348. Design Tic-Tac-Toe](https://leetcode.com/problems/design-tic-tac-toe/)

这道题的描述是，设计一个井字棋游戏，这个游戏有两个玩家，玩家轮流在 3x3 的棋盘上放置棋子，玩家可以放置自己的棋子，也可以放置对方的棋子，玩家获胜的条件是，如果玩家在一行，一列或者对角线上放置了三个棋子，玩家就获胜了。

test cases:

```text
Input
["TicTacToe", "move", "move", "move", "move", "move", "move", "move"]
[[3], [0, 0, 1], [0, 2, 2], [2, 2, 1], [1, 1, 2], [2, 0, 1], [1, 0, 2], [2, 1, 1]]
Output
[null, 0, 0, 0, 0, 0, 0, 1]

Explanation
TicTacToe ticTacToe = new TicTacToe(3);
Assume that player 1 is "X" and player 2 is "O" in the board.
ticTacToe.move(0, 0, 1); // return 0 (no one wins)
|X| | |
| | | |    // Player 1 makes a move at (0, 0).
| | | |

ticTacToe.move(0, 2, 2); // return 0 (no one wins)
|X| |O|
| | | |    // Player 2 makes a move at (0, 2).
| | | |

ticTacToe.move(2, 2, 1); // return 0 (no one wins)
|X| |O|
| | | |    // Player 1 makes a move at (2, 2).
| | |X|

ticTacToe.move(1, 1, 2); // return 0 (no one wins)
|X| |O|
| |O| |    // Player 2 makes a move at (1, 1).
| | |X|

ticTacToe.move(2, 0, 1); // return 0 (no one wins)
|X| |O|
| |O| |    // Player 1 makes a move at (2, 0).
|X| |X|

ticTacToe.move(1, 0, 2); // return 0 (no one wins)
|X| |O|
|O|O| |    // Player 2 makes a move at (1, 0).
|X| |X|

ticTacToe.move(2, 1, 1); // return 1 (player 1 wins)
|X| |O|
|O|O| |    // Player 1 makes a move at (2, 1).
|X|X|X|
```

这道题的思路是，我们使用两个列表来存储玩家在每一行和每一列上的棋子数，使用两个变量来存储玩家在对角线上的棋子数，每次玩家放置棋子的时候，我们就更新这些数据，如果玩家在某一行，某一列或者某一对角线上放置了三个棋子，玩家就获胜了。

这里我们就不赘述三个 check 的代码，因为实在是太简单了，我们直接用 O(1)的时间复杂度来完成这道题。

```python
class TicTacToe(object):
    def __init__(self, n):
        self.row = [0] * n
        self.col = [0] * n
        self.diag = 0
        self.anti_diag = 0
        self.length = n

    def move(self, row, col, player):
        offset = player * 2 - 3 # player 1 -> -1, player 2 -> 1
        self.row[row] += offset
        self.col[col] += offset
        if row == col:
            self.diag += offset
        if row + col == self.length - 1:
            self.anti_diag += offset
        if self.length in [self.row[row], self.col[col], self.diag, self.anti_diag]:
            return 2
        if -self.length in [self.row[row], self.col[col], self.diag, self.anti_diag]:
            return 1
        return 0
```

### 高级题

#### [460. LFU Cache](https://leetcode.com/problems/lfu-cache/)

这道题的描述是，设计一个 LFU Cache，这个 Cache 支持 get 和 put 操作，其中 get 操作是获取 key 对应的 value，如果 key 不存在，返回 -1，put 操作是插入 key 和 value，如果 key 已经存在，更新 value，如果 key 不存在，插入 key 和 value，如果 Cache 已经满了，删除访问次数最少的 key，如果有多个 key 的访问次数相同，删除最早访问的 key。

test cases:

```text
Input
["LFUCache", "put", "put", "get", "put", "get", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, 3, null, -1, 3, 4]

Explanation
// cnt(x) = the use counter for key x
// cache=[] will show the last used order for tiebreakers (leftmost element is  most recent)
LFUCache lfu = new LFUCache(2);
lfu.put(1, 1);   // cache=[1,_], cnt(1)=1
lfu.put(2, 2);   // cache=[2,1], cnt(2)=1, cnt(1)=1
lfu.get(1);      // return 1
                 // cache=[1,2], cnt(2)=1, cnt(1)=2
lfu.put(3, 3);   // 2 is the LFU key because cnt(2)=1 is the smallest, invalidate 2.
                 // cache=[3,1], cnt(3)=1, cnt(1)=2
lfu.get(2);      // return -1 (not found)
lfu.get(3);      // return 3
                 // cache=[3,1], cnt(3)=2, cnt(1)=2
lfu.put(4, 4);   // Both 1 and 3 have the same cnt, but 1 is LRU, invalidate 1.
                 // cache=[4,3], cnt(4)=1, cnt(3)=2
lfu.get(1);      // return -1 (not found)
lfu.get(3);      // return 3
                 // cache=[3,4], cnt(4)=1, cnt(3)=3
lfu.get(4);      // return 4
                 // cache=[4,3], cnt(4)=2, cnt(3)=3
```

这道题的思路是，我们使用两个哈希表，一个哈希表用来存储 key 和 value，另一个哈希表用来存储 key 和对应的访问次数，我们使用一个双向链表来存储访问次数相同的 key，这样我们就可以在 O(1)的时间复杂度内完成 get 和 put 操作。

```python
import collections

class Node:
    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.freq = 1
        self.prev = None
        self.next = None

class LinkedList:
    def __init__(self):
        self.size = 0
        self.cache = {}
        self.dummy_left = Node()
        self.dummy_right = Node()
        self.dummy_left.next = self.dummy_right
        self.dummy_left.prev = self.dummy_right
        self.dummy_right.next = self.dummy_left
        self.dummy_right.prev = self.dummy_left

    def __len__(self):
        return self.size

    def add_element(self, node: Node):
        last_element = self.dummy_right.prev
        self.dummy_right.prev = node
        node.next = self.dummy_right
        last_element.next = node
        node.prev = last_element
        self.size += 1

    def remove_element(self, node: Node):
        node.next.prev = node.prev
        node.prev.next = node.next
        self.size -= 1

    def pop_element(self, node: Node=None):
        if self.size == 0:
            return

        if not node:
            node = self.dummy_left.next
            self.remove_element(node)
        else:
            self.remove_element(node)
        return node

class LFUCache:
    def __init__(self, capacity):
        self._size = 0
        self._capacity = capacity

        self._node = dict() # key: Node
        self._freq = collections.defaultdict(LinkedList)
        self._minfreq = 0

    def _update(self, node):
        freq = node.freq
        self._freq[freq].pop_element(node)
        if self._minfreq == freq and not self._freq[freq]:
            self._minfreq += 1
        node.freq += 1
        freq = node.freq
        self._freq[freq].add_element(node)

    def get(self, key):
        if key not in self._node:
            return -1
        node = self._node[key]
        self._update(node)
        return node.val

    def put(self, key, value):
        if self._capacity == 0:
            return

        if key in self._node:
            node = self._node[key]
            self._update(node)
            node.val = value
        else:
            if self._size == self._capacity:
                node = self._freq[self._minfreq].pop_element()
                del self._node[node.key]
                self._size -= 1

            node = Node(key, value)
            self._node[key] = node
            self._freq[1].add_element(node)
            self._minfreq = 1
            self._size += 1
```

#### [716. Max Stack](https://leetcode.com/problems/max-stack/)

这道题的描述是，设计一个栈，这个栈支持 push、pop、top、peekMax 和 popMax 操作，其中 push、pop、top 操作和普通栈一样，peekMax 操作是获取栈中最大的元素，popMax 操作是删除栈中最大的元素，如果有多个最大的元素，删除最靠近栈顶的那个。

test cases:

```text
Input
["MaxStack", "push", "push", "push", "top", "popMax", "top", "peekMax", "pop", "top"]
[[], [5], [1], [5], [], [], [], [], [], []]
Output
[null, null, null, null, 5, 5, 1, 5, 1, 5]

Explanation
MaxStack stk = new MaxStack();
stk.push(5);   // [5] the top of the stack and the maximum number is 5.
stk.push(1);   // [5, 1] the top of the stack is 1, but the maximum is 5.
stk.push(5);   // [5, 1, 5] the top of the stack is 5, which is also the maximum, because it is the top most one.
stk.top();     // return 5, [5, 1, 5] the stack did not change.
stk.popMax();  // return 5, [5, 1] the stack is changed now, and the top is different from the max.
stk.top();     // return 1, [5, 1] the stack did not change.
stk.peekMax(); // return 5, [5, 1] the stack did not change.
stk.pop();     // return 1, [5] the top of the stack and the max element is now 5.
stk.top();     // return 5, [5] the stack did not change.
```

这道题的思路是，我们使用两个栈，一个栈用来存储数据，另一个栈用来存储当前栈中的最大值，这样我们就可以在 O(1) 的时间复杂度内完成所有操作。

```python
class MaxStack:
    def __init__(self):
        self.stack = []
        self.max_stack = []

    def push(self, x):
        self.stack.append(x)
        if not self.max_stack or x >= self.max_stack[-1]:
            self.max_stack.append(x)

    def pop(self):
        x = self.stack.pop()
        if x == self.max_stack[-1]:
            self.max_stack.pop()
        return x

    def top(self):
        return self.stack[-1]

    def peekMax(self):
        return self.max_stack[-1]

    def popMax(self):
        x = self.max_stack.pop()
        buffer = []
        while self.stack[-1] != x:
            buffer.append(self.stack.pop())

        self.stack.pop()
        for item in reversed(buffer):
            self.push(item)

        return x
```

然而这样做会 TLE，我们需要用 log（n）的时间复杂度来完成 popMax 操作，我们可以使用一个优先队列来存储数据，这样我们就可以在 O(logn) 的时间复杂度内完成 popMax 操作。

```python
import heapq

class MaxStack:

    def __init__(self):
        self.heap = []
        self.cnt = 0
        self.stack = []
        self.removed = set()

    def push(self, x: int) -> None:
        heapq.heappush(self.heap, (-x, -self.cnt))
        self.stack.append((x, self.cnt))
        self.cnt += 1

    def pop(self) -> int:
        while self.stack and self.stack[-1][1] in self.removed:
            self.stack.pop()
        num, idx = self.stack.pop()
        self.removed.add(idx)
        return num

    def top(self) -> int:
        while self.stack and self.stack[-1][1] in self.removed:
            self.stack.pop()
        return self.stack[-1][0]

    def peekMax(self) -> int:
        while self.heap and -self.heap[0][1] in self.removed:
            heapq.heappop(self.heap)
        return -self.heap[0][0]

    def popMax(self) -> int:
        while self.heap and -self.heap[0][1] in self.removed:
            heapq.heappop(self.heap)
        num, idx = heapq.heappop(self.heap)
        self.removed.add(-idx)
        return -num
```
