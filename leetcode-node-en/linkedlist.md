LRU

```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None
        
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def add_node(self, node):
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def move_to_head(self, node):
        self.remove_node(node)
        self.add_node(node)
    
    def pop_tail(self):
        res = self.tail.prev
        self.remove_node(res)
        return res
    
    def get(self, key: int) -> int:
        node = self.cache.get(key)
        if not node:
            return -1
        
        self.move_to_head(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key)
        
        if not node:
            newNode = Node()
            newNode.key = key
            newNode.val = value
            self.cache[key] = newNode
            self.add_node(newNode)
            
            if len(self.cache) > self.capacity:
                tail = self.pop_tail()
                del self.cache[tail.key]
        else:
            node.val = value

```