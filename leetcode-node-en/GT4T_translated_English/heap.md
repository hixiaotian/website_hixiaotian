### Foreword

In this topic, we will focus on the specific usage of heap.

There are many types of questions that use heap, such as top K questions, merging K ordered linked lists, merging K ordered arrays, and so on.

Why use them? Because they can help us find the maximum or minimum value quickly.

So, when we look at some of the words of the problem, such as top K, kth big, kth small, we can think of heap.

### What is heap?

A heap is a data structure that can find the maximum or minimum in O (1) time.

In python, we can use heapq to implement heap. Next, we will explain the common methods of heap.

Note that heapq is a small top heap, and if we need a large top heap, we need to take the first element negative and insert it into the heap.

```python
import heapq

# 初始化一个 heap
heap = []

# 往 heap 里插入一个元素，然后 heap 会自动调整，复杂度为 O(logn)
heapq.heappush(heap, 1)

# 从 heap 里弹出最小的元素，复杂度为 O(logn)
heapq.heappop(heap)

# 从 heap 里取出最小的元素，但是不弹出，复杂度为 O(1)
heap[0]

# 把 heap 的最小的元素替换成 newvalue，然后弹出最小的元素，然后 heap 会自动调整，复杂度为 O(logn)
heapq.heapreplace(heap, newvalue)

# 把 heap 先push一个 newitem，然后再弹出最小的元素，复杂度为 O(logn)
heapq.heappushpop(heap, newitem)

# heapreplace 和 heappushpop的区别是，当 newvalue < heap[0] 时，heapreplace会先弹出最小的元素，然后再插入 newvalue，而 heappushpop 会直接返回 newvalue，不会插入 newvalue。

# 取 heap 中最大的 n 个元素，复杂度为 O(nlogn)
heapq.nlargest(n, heap)

# 取 heap 中最小的 n 个元素，复杂度为 O(nlogn)
heapq.nsmallest(n, heap)
```

Next, let's look at some examples!

#### [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

The description of this problem is to find the kth largest element in the array.

test cases:

```text
Input: [3,2,1,5,6,4] and k = 2
Output: 5

Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
```

The idea of this problem is that we can use a heap of size K, and then traverse the array. If the current element is larger than the first element of the heap, it means that the first element of the heap is not the kth largest element. We can pop out the first element, and then insert the current element into the heap.

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            if len(heap) < k:
                heapq.heappush(heap, num)
            else:
                if num > heap[0]:
                    heapq.heappushpop(heap, num)
                    # 这里也可以使用 heapq.heappushpop(heap, num)，区别是 heapreplace 会先弹出最小的元素，然后再插入 num，而 heappushpop 会直接返回 num，不会插入 num。
        return heap[0]
```

Of course, we can also use heapq. Nlargest to solve this problem.

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k, nums)[-1]
```

You can also think in reverse and use the maximum heap to solve this problem.

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = [-num for num in nums]
        heapq.heapify(heap)
        for _ in range(k):
            res = heapq.heappop(heap)
        return -res
```

Complexity analysis:

- Time complexity: O (n log K). The reason is that we need to traverse the entire array, and then each insertion requires a time complexity of log K.
- Space complexity: O (K), because we need to use the heap to store numbers.

#### [378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

The description of this problem is to find the kth smallest element in an ordered matrix.

test cases:

```text
Input:
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8

Output: 13
```

The idea of this problem is that we can use a heap of size K, then traverse the matrix, put the first element of each matrix into the heap, then take the smallest element from the heap each time, and then put the next element of the matrix in which this element is located into the heap.

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        heap = []
        for i in range(len(matrix)):
            heapq.heappush(heap, (matrix[i][0], i, 0)) # 先把每个矩阵的第一个元素放入 heap, 同时记录这个元素所在矩阵的坐标
        res = 0
        for _ in range(k): # 然后从 heap 中取出最小的元素，然后把这个元素所在矩阵的下一个元素放入 heap
            res, i, j = heapq.heappop(heap) # 这里的 i 和 j 是矩阵的坐标
            if j + 1 < len(matrix[0]): # 如果 j + 1 < len(matrix[0])，说明这个元素不是所在矩阵的最后一个元素，我们可以把这个元素所在矩阵的下一个元素放入 heap
                heapq.heappush(heap, (matrix[i][j + 1], i, j + 1))
        return res
```

Complexity analysis:

- Time complexity: O (K log K). The reason is that we need to traverse the entire matrix, and then each insertion requires time complexity of log K.
- Space complexity: O (K), because we need to use the heap to store numbers.

#### [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)

The description of this problem is to find the K elements with the highest frequency in the array.

test cases:

```text
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Input: nums = [1], k = 1
Output: [1]
```

The idea here is that we can take a heap of size K, and we can go through the array, put each element into the hash table, and then put the elements in the hash table into the heap, and then take the smallest element out of the heap each time, Then put the next element of the hash table where this element is located into the heap.

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        heap = []
        count = collections.Counter(nums)
        for num, freq in count.items():
            if len(heap) < k:
                heapq.heappush(heap, (freq, num))
            else:
                if freq > heap[0][0]:
                    heapq.heappushpop(heap, (freq, num))
        return [num for freq, num in heap]
```

You can also use heapq. Nlargest to solve the problem.

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = collections.Counter(nums)
        return heapq.nlargest(k, count.keys(), key=count.get)
```

Of course, you can also use the maximum heap to solve this problem.

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        heap = []
        count = collections.Counter(nums)
        for num, freq in count.items():
            heapq.heappush(heap, (-freq, num))
        res = []
        for _ in range(k):
            res.append(heapq.heappop(heap)[1])
        return res
```

Complexity analysis:

- Time complexity: O (n log K). The reason is that we need to traverse the entire array, and then each insertion requires a time complexity of log K.
- Space complexity: O (K), because we need to use the heap to store numbers.

#### [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)

The description of this problem is to find the K elements with the highest frequency in the array.

test cases:

```text
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]

Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]
```

The idea here is that we can take a heap of size K, and we can go through the array, put each element into the hash table, and then put the elements in the hash table into the heap, and then take the smallest element out of the heap each time, Then put the next element of the hash table where this element is located into the heap.

！！ Note that this problem requires the smallest K elements to be returned in alphabetical order, and it needs to be sorted from large to small. So we can only use the maximum heap to solve this problem.

```python
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        heap = []
        count = collections.Counter(words)
        for word, freq in count.items():
            heapq.heappush(heap, (-freq, word))
        res = []
        for _ in range(k):
            res.append(heapq.heappop(heap)[1])
        return res
```

Complexity analysis:

- Time complexity: O (n log K). The reason is that we need to traverse the entire array, and then each insertion requires a time complexity of log K.
- Space complexity: O (K), because we need to use the heap to store numbers.

#### [253. meeting rooms ii](https://leetcode.com/problems/meeting-rooms-ii/)

The description of this problem is to find the minimum number of meeting rooms given the start and end time of a group of meetings.

test cases:

```text
Input: [[0, 30],[5, 10],[15, 20]]
Output: 2

Input: [[7,10],[2,4]]
Output: 1
```

The idea of this problem is that we sort the interval according to the start time, then use a heap to store the end time, and then traverse the array. If the current start time is greater than the minimum value of the heap, then pop out the minimum value of the heap, and then put the current end time into the heap. The reason for this is that if the current start time is greater than the minimum value of the heap, then the meeting can use the previous meeting room, so we pop out the end time of the previous meeting room, and then put the current end time into the heap. This is a greedy algorithm that changes direction.

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort()
        heap = []
        heapq.heappush(heap, intervals[0][1]) # 把第一个会议的结束时间放入 heap
        for i in range(1, len(intervals)):
            if intervals[i][0] >= heap[0]: # 如果当前的开始时间大于 heap 的最小值，那么就把 heap 的最小值 pop 出来，然后把当前的结束时间放入 heap
                heapq.heappop(heap) # 仔细想一下，其实就相当于抵消了一个会议室
            heapq.heappush(heap, intervals[i][1]) # 把当前的结束时间放入 heap
        return len(heap)
```

Complexity analysis:

- Time complexity: O (nlogn). The reason is that we need to traverse the entire array, and then each insertion requires logn time complexity.
- Space complexity: O (n), because we need to use the heap to store numbers.

#### [373. Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)

The description of this problem is to find the K logarithm with the smallest sum in two arrays.

test cases:

```text

Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
Output: [[1,2],[1,4],[1,6]]

Input: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
Output: [1,1],[1,1]
```

The idea of this problem is that we can use a heap of size K, then go through the array, put all the combinations into the heap, and then take the smallest element out of the heap each time.

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        heap = []
        for num1 in nums1:
            for num2 in nums2:
                heapq.heappush(heap, (num1 + num2, [num1, num2]))
        res = []
        for _ in range(min(k, len(heap))):
            res.append(heapq.heappop(heap)[1])
        return res
```

But it's going to time out, because we put all the combinations into the heap, but we only need the smallest K combinations, so we can use a heap of size K, then go through the array, put the first element of each array into the heap, and then take the smallest element out of the heap each time. The next element in the array is then placed in the heap.

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        res = []
        visited = set()

        min_heap = [(nums1[0] + nums2[0], (0, 0))]

        visited.add((0, 0))
        count = 0

        while k > 0 and min_heap:
            val, (i, j) = heapq.heappop(min_heap)
            res.append([nums1[i], nums2[j]])

            if i + 1 < len(nums1) and (i + 1, j) not in visited:
                heapq.heappush(min_heap, (nums1[i + 1] + nums2[j], (i + 1, j)))
                visited.add((i + 1, j))

            if j + 1 < len(nums2) and (i, j + 1) not in visited:
                heapq.heappush(min_heap, (nums1[i] + nums2[j + 1], (i, j + 1)))
                visited.add((i, j + 1))

            k -= 1

        return res
```

Complexity analysis:

- Time complexity: O (K log K). The reason is that we need to traverse the entire array, and then each insertion requires log K time complexity.
- Space complexity: O (K), because we need to use the heap to store numbers.

#### [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

The description of this problem is to merge K ordered linked lists.

test cases:

```text
Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6
```

The idea of this problem is that we can use a heap of size K, then go through the linked list, put the first element of each linked list into the heap, then take the smallest element from the heap each time, and then put the next element of the linked list where the element is in into the heap.

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        heap = []
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(heap, (node.val, i))
        dummy = ListNode(0)
        curr = dummy
        while heap:
            val, i = heapq.heappop(heap)
            curr.next = lists[i]
            curr = curr.next
            lists[i] = lists[i].next
            if lists[i]:
                heapq.heappush(heap, (lists[i].val, i))
        return dummy.next
```

Complexity analysis:

- Time complexity: O (n log K). The reason is that we need to traverse the entire linked list, and then each insertion requires a time complexity of log K.
- Space complexity: O (K), because we need to use the heap to store numbers.

#### [407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/)

The description of this problem is to give a two-dimensional array, the number in the array represents the height, and then ask how much water the two-dimensional array can store.

test cases:

```text
Given the following 3x6 height map:
[
  [1,4,3,1,3,2],
  [3,2,1,3,2,4],
  [2,3,3,2,3,1]
]

Return 4.
```

The idea of this problem is that we can use a heap of size K, then traverse the four sides of the matrix, put the elements of each side into the heap, then take the smallest element from the heap each time, and then put the next element of the matrix in which this element is located into the heap.

```python
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        if not heightMap or not heightMap[0]:
            return 0

        heap = []
        visited = set()

        m, n = len(heightMap), len(heightMap[0])

        for i in range(m):
            heapq.heappush(heap, (heightMap[i][0], i, 0))
            heapq.heappush(heap, (heightMap[i][n - 1], i, n - 1))
            visited.add((i, 0))
            visited.add((i, n - 1))

        for j in range(n):
            heapq.heappush(heap, (heightMap[0][j], 0, j))
            heapq.heappush(heap, (heightMap[m - 1][j], m - 1, j))
            visited.add((0, j))
            visited.add((m - 1, j))

        res = 0
        while heap:
            height, i, j = heapq.heappop(heap)
            for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                if 0 <= x < m and 0 <= y < n and (x, y) not in visited:
                    res += max(0, height - heightMap[x][y])
                    heapq.heappush(heap, (max(height, heightMap[x][y]), x, y))
                    visited.add((x, y))

        return res
```

Complexity analysis:

- Time complexity: O (Mn log (Mn)). The reason is that we need to traverse the entire matrix and then log (Mn) time complexity for each insertion.
- Space complexity: O (Mn), because we need to use heap to store numbers.

#### [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
