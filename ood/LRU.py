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

    def put_to_right(self, node):
        self.remove_element(node)
        self.add_element(node)

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self.put_to_right(node)
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
