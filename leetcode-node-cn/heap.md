### 前言

在这个专题，我们将主要讨论 heap 的具体用法。

有很多类型题会使用到 heap, 比如 top k 问题，合并 k 个有序链表，合并 k 个有序数组等等。

为什么使用他们呢？因为他们可以帮助我们快速的找到最大值或者最小值。

所以，在我们看到问题的一些字眼时，比如 top k，第 k 大，第 k 小，我们就可以想到 heap。

### 什么是 heap

heap 是一种数据结构，它的特点是可以在 O(1)的时间复杂度内找到最大值或者最小值。

在 python 里，我们可以使用 heapq 来实现 heap。我们接下来会讲解 heap 的常用方法。

注意，heapq 是小顶堆，如果我们需要大顶堆，我们需要把第一个元素取负数，然后再插入 heap。

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

接下来我们看几道例题！

#### [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

这道题的描述是找到数组中第 k 大的元素。

test cases:

```text
Input: [3,2,1,5,6,4] and k = 2
Output: 5

Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
```

这道题的思路是，我们可以使用一个大小为 k 的 heap，然后遍历数组，如果当前元素比 heap 的第一个元素大，说明 heap 的第一个元素不是第 k 大的元素，我们可以把第一个元素弹出，然后把当前元素插入 heap。

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            if len(heap) < k:
                heapq.heappush(heap, num)
            else:
                if num > heap[0]:
                    heapq.heapreplace(heap, num)
                    # 这里也可以使用 heapq.heappushpop(heap, num)，区别是 heapreplace 会先弹出最小的元素，然后再插入 num，而 heappushpop 会直接返回 num，不会插入 num。
        return heap[0]
```

当然，我们也可以使用 heapq.nlargest 来解决这道题。

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k, nums)[-1]
```

复杂度分析：

- 时间复杂度：O(nlogk)，原因是我们需要遍历整个数组，然后每次插入都需要 logk 的时间复杂度。
- 空间复杂度：O(k)，原因是我们需要使用 heap 来存储数字。

#### [378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

这道题的描述是找到一个有序矩阵中第 k 小的元素。

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

这道题的思路是，我们可以使用一个大小为 k 的 heap，然后遍历矩阵，把每个矩阵的第一个元素放入 heap，然后每次从 heap 中取出最小的元素，然后把这个元素所在矩阵的下一个元素放入 heap。

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        heap = []
        for i in range(len(matrix)):
            heapq.heappush(heap, (matrix[i][0], i, 0))
        res = 0
        for _ in range(k):
            res, i, j = heapq.heappop(heap)
            if j + 1 < len(matrix[0]):
                heapq.heappush(heap, (matrix[i][j + 1], i, j + 1))
        return res
```

复杂度分析：

- 时间复杂度：O(klogk)，原因是我们需要遍历整个矩阵，然后每次插入都需要 logk 的时间复杂度。
- 空间复杂度：O(k)，原因是我们需要使用 heap 来存储数字。

#### [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)

这道题的描述是找到数组中出现频率最高的 k 个元素。

test cases:

```text
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Input: nums = [1], k = 1
Output: [1]
```

这道题的思路是，我们可以使用一个大小为 k 的 heap，然后遍历数组，把每个元素放入 hash table，然后把 hash table 中的元素放入 heap，然后每次从 heap 中取出最小的元素，然后把这个元素所在 hash table 的下一个元素放入 heap。

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
                    heapq.heapreplace(heap, (freq, num))
        return [num for freq, num in heap]
```

复杂度分析：

- 时间复杂度：O(nlogk)，原因是我们需要遍历整个数组，然后每次插入都需要 logk 的时间复杂度。
- 空间复杂度：O(k)，原因是我们需要使用 heap 来存储数字。

#### [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)

这道题的描述是找到数组中出现频率最高的 k 个元素。

test cases:

```text
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]

Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]
```

这道题的思路是，我们可以使用一个大小为 k 的 heap，然后遍历数组，把每个元素放入 hash table，然后把 hash table 中的元素放入 heap，然后每次从 heap 中取出最小的元素，然后把这个元素所在 hash table 的下一个元素放入 heap。

```python
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        heap = []
        count = collections.Counter(words)
        for word, freq in count.items():
            if len(heap) < k:
                heapq.heappush(heap, (-freq, word))
            else:
                if (-freq, word) > heap[0]:
                    heapq.heapreplace(heap, (-freq, word))
        return [word for freq, word in heap]
```

复杂度分析：

- 时间复杂度：O(nlogk)，原因是我们需要遍历整个数组，然后每次插入都需要 logk 的时间复杂度。
- 空间复杂度：O(k)，原因是我们需要使用 heap 来存储数字。

#### [253. meeting rooms ii](https://leetcode.com/problems/meeting-rooms-ii/)

这道题的描述是给定一组会议的开始和结束时间，求最少需要多少个会议室。

test cases:

```text
Input: [[0, 30],[5, 10],[15, 20]]
Output: 2

Input: [[7,10],[2,4]]
Output: 1
```

这道题的思路是，我们可以使用一个大小为 k 的 heap，然后遍历数组，把每个元素放入 heap，然后每次从 heap 中取出最小的元素，然后把这个元素所在 hash table 的下一个元素放入 heap。

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[0])
        heap = []
        heapq.heappush(heap, intervals[0][1])
        for i in range(1, len(intervals)):
            if intervals[i][0] >= heap[0]:
                heapq.heappop(heap)
            heapq.heappush(heap, intervals[i][1])
        return len(heap)
```

复杂度分析：

- 时间复杂度：O(nlogn)，原因是我们需要遍历整个数组，然后每次插入都需要 logn 的时间复杂度。
- 空间复杂度：O(n)，原因是我们需要使用 heap 来存储数字。

#### [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

这道题的描述是合并 k 个有序链表。

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

这道题的思路是，我们可以使用一个大小为 k 的 heap，然后遍历链表，把每个链表的第一个元素放入 heap，然后每次从 heap 中取出最小的元素，然后把这个元素所在链表的下一个元素放入 heap。

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

复杂度分析：

- 时间复杂度：O(nlogk)，原因是我们需要遍历整个链表，然后每次插入都需要 logk 的时间复杂度。
- 空间复杂度：O(k)，原因是我们需要使用 heap 来存储数字。

#### [373. Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)

这道题的描述是找到两个数组中和最小的 k 对数。

test cases:

```text

Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
Output: [[1,2],[1,4],[1,6]]

Input: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
Output: [1,1],[1,1]
```

这道题的思路是，我们可以使用一个大小为 k 的 heap，然后遍历数组，把所有的组合放入 heap，然后每次从 heap 中取出最小的元素。

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

但是这么做会超时，原因是我们把所有的组合都放入了 heap，但是我们只需要最小的 k 个组合，所以我们可以使用一个大小为 k 的 heap，然后遍历数组，把每个数组的第一个元素放入 heap，然后每次从 heap 中取出最小的元素，然后把这个元素所在数组的下一个元素放入 heap。

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

复杂度分析：

- 时间复杂度：O(klogk)，原因是我们需要遍历整个数组，然后每次插入都需要 logk 的时间复杂度。
- 空间复杂度：O(k)，原因是我们需要使用 heap 来存储数字。

#### [407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/)

这道题的描述是给定一个二维数组，数组中的数字代表高度，然后问这个二维数组能够存储多少水。

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

这道题的思路是，我们可以使用一个大小为 k 的 heap，然后遍历矩阵的四条边，把每个边的元素放入 heap，然后每次从 heap 中取出最小的元素，然后把这个元素所在矩阵的下一个元素放入 heap。

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

复杂度分析：

- 时间复杂度：O(mnlog(mn))，原因是我们需要遍历整个矩阵，然后每次插入都需要 log(mn) 的时间复杂度。
- 空间复杂度：O(mn)，原因是我们需要使用 heap 来存储数字。

#### [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
