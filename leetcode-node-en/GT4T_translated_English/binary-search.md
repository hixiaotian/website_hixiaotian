### Foreword

In the dichotomy, you need to master the template. Here are two templates, one is the standard template, and the other is the undead template. The standard template is the answer provided in most leetcode templates, while the undead template is the "never-ending loop template" from the nine-chapter algorithm, each with its own advantages.

Of course, python also provides bisect package, but in the interview, it is not recommended to use, but for OA problem solving, it is recommended to use this, because it saves time and effort.

Next, take searching for a target in an ordered array as an example to compare the differences.

<b>↓ Click on the title to jump directly to the leetcode title page ↓</b>

### Basic questions

#### [704. Binary Search](https://leetcode.com/problems/binary-search/)

test cases:

```text
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
```

Standard template (notes noted in code): ![二分法基本图解](https://assets.leetcode.com/static_assets/posts/1EYkSkQaoduFBhpCVx7nyEA.gif)

```python
def search(self, nums: List[int], target: int) -> int:
    if not nums:
        return -1

    left, right = 0, len(nums) - 1
    while left <= right:  # 这里一定要 <=, left 和 right是允许重合的
        mid = left + (right - left) // 2  # 防止累加爆上限
        if nums[mid] == target:
            return mid
        if target < nums[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return -1
```

In the undead loop template, we slightly modify the condition of while (notes have been noted in the code):

```python
def search(self, nums: List[int], target: int) -> int:
    if not nums:
        return -1

    left, right = 0, len(nums) - 1
    while left + 1 < right:  # 这里变成了 +1 <, 相当于在离1个的时候就停止了
        mid = left + (right - left) // 2  # 防止累加爆上限
        if nums[mid] > target:
            right = mid
        elif nums[mid] < target:
            left = mid
        else:
            return mid

    if nums[left] == target: # 最后对接近的left 做判断
        return left

    if nums[right] == target: # 最后对接近的right 做判断
        return right

    return -1
```

Direct use of the bisect package is convenient and convenient, but it is not recommended to use it in the interview:

```python
from bisect import bisect_left
def search(self, nums: List[int], target: int) -> int:
    num_size = len(nums)
    i = bisect_left(nums, target)
    return i if i < num_size and nums[i] == target else -1
```

Time complexity: O (log N) Space complexity: O (1)

#### [35. Search Insert Position](https://leetcode.com/problems/search-insert-position)

The difference between this question and the previous one is that he has to decide where to insert, so he only needs to modify the final condition.

test cases:

```text
Input: nums = [1,3,5,6], target = 5
Output: 2

Input: nums = [1,3,5,6], target = 2
Output: 1

Input: nums = [1,3,5,6], target = 7
Output: 4
```

```python
# 1. 标准模版
def searchInsert(self, nums: List[int], target: int) -> int:
    if not nums:
        return -1

    left, right = 0, len(nums) - 1
    while left <= right:  # 这里一定要 <=, left 和 right是允许重合的
        mid = left + (right - left) // 2  # 防止累加爆上限
        if nums[mid] == target:
            return mid
        if target < nums[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return left


# 2. 不死模版
def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        mid = (left + right) // 2

        while left + 1 < right:

            mid = (left + right) // 2
            print(left, right, mid)
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid
            else:
                left = mid

        if nums[right] < target: # 这里必须先使用right，想想为什么？
            return right + 1

        if nums[left] < target:
            return left + 1

        return left


# 3. bisect 包
def searchInsert(self, nums: List[int], target: int) -> int:
    return bisect.bisect_left(nums, target)
```

Solution complexity analysis: Time complexity: O (log N) Space complexity: O (1)

#### [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

This question involves repeating elements, so pay attention to where to stop? Modify the if condition to test cases:

```text
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]

Input: nums = [], target = 0
Output: [-1,-1]
```

```python
# 1. 标准模版
def searchRange(self, nums: List[int], target: int) -> List[int]:
    def findLeft(nums, target):
        res = -1
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] >= target: # 把等于的情况，给到右边，这样就能找到左边界
                if nums[mid] == target:
                    res = mid
                end = mid - 1
            else:
                start = mid + 1
        return res

    def findRight(nums, target):
        res = -1
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] <= target: # 把等于的情况，给到左边，这样就能找到右边界
                if nums[mid] == target:
                    res = mid
                start = mid + 1
            else:
                end = mid - 1
        return res

    if nums == None or len(nums) == 0:
        return [-1, -1]
    return [findLeft(nums, target), findRight(nums,target)]


# 2. 不死模版
def searchRange(self, nums: List[int], target: int) -> List[int]:
    def findLeft(nums, target):
        start, end = 0, len(nums) - 1

        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] >= target: # 把等于的情况，给到右边，这样就能找到左边界
                end = mid
            else:
                start = mid

        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        return -1

    def findRight(nums, target):
        start, end = 0, len(nums) - 1

        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] > target:
                end = mid
            else: # 把等于的情况，给到左边，这样就能找到右边界
                start = mid

        if nums[end] == target: # 这种情况之下为什么要先判断end呢？
            return end
        if nums[start] == target:
            return start
        return -1

    if nums == None or len(nums) == 0:
        return [-1, -1]
    return [findLeft(nums, target), findRight(nums,target)]

# 3. bisect 包
def searchRange(self, nums: List[int], target: int) -> List[int]:
    if not nums:
        return [-1, -1]
    left,right = bisect.bisect_left(nums, target), bisect.bisect_right(nums, target)
    return [left, right - 1] if left < right else [-1, -1]
```

Solution complexity analysis: Time complexity: O (log N) Space complexity: O (1)

#### [702. Search in a Sorted Array of Unknown Size](https://leetcode.com/problems/search-in-a-sorted-array-of-unknown-size/)

Test cases:

```text
Input: secret = [-1,0,3,5,9,12, ...], target = 9
Output: 4
Explanation: 9 exists in secret and its index is 4.

Input: secret = [-1,0,3,5,9,12, ...], target = 2
Output: -1
Explanation: 2 does not exist in secret so return -1.

```

This basic idea is the same as the previous one. There is only one judgment point, so we only need to add one place. The dichotomy solution is the same as the previous one!

```python
def search(self, reader, target):
    start, end = 0, 1
    # 比target小，就疯狂让end 以次方级别增加
    while reader.get(end) < target:
        end <<= 1

    # 这里用之前的二分解法就可以了，思路不变
    ...
```

#### [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)

Test cases:

```text
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true

Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false
```

In this problem, we only need to look at the two-dimensional array as a one-dimensional array, and then it is a standard binary search.

```python
1. 标准模版
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    m = len(matrix)
    if m == 0:
        return False
    n = len(matrix[0])

    left, right = 0, m * n - 1
    while left <= right:
        pivot_idx = (left + right) // 2
        # 这里我们根据每一行的长度，来计算出这个pivot_idx在二维数组中的位置
        pivot_element = matrix[pivot_idx // n][pivot_idx % n]
        if target == pivot_element:
            return True
        elif target < pivot_element:
            right = pivot_idx - 1
        else:
            left = pivot_idx + 1
    return False

# 2. 不死模版
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    m = len(matrix)
    if m == 0:
        return False
    n = len(matrix[0])

    left, right = 0, m * n - 1
    while left + 1 < right:
        pivot_idx = (left + right) // 2
        # 这里我们根据每一行的长度，来计算出这个pivot_idx在二维数组中的位置
        pivot_element = matrix[pivot_idx // n][pivot_idx % n]
        if target == pivot_element:
            return True
        elif target < pivot_element:
            right = pivot_idx
        else:
            left = pivot_idx

    if matrix[left // n][left % n] == target:
        return True
    if matrix[right // n][right % n] == target:
        return True
    return False

# 3. bisect 包
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    import bisect

    if target < matrix[0][0] or target > matrix[-1][-1]:
        return False

    first_column = [matrix[i][0] for i in range(len(matrix))]

    row = bisect.bisect(first_column, target) - 1
    if matrix[row][0] == target:
        return True

    col = bisect.bisect(matrix[row], target) - 1
    if matrix[row][col] == target:
        return True

    return False
```

### Variant problem

#### [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

Test cases:

```text
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Input: nums = [1], target = 0
Output: -1
```

Train of thought: This problem is also just a variant, just need to judge, if nums [start] < nums[end]，那么就是有序的，直接返回 nums[start]就可以了。如果不是有序的，那么就是旋转过的，那么我们就需要判断一下，如果 nums[mid] > nums [mid-1] and nums [mid] > nums (mid + 1), then nums (mid + 1) is the minimum value. If not, then we need to judge that if nums [mid] > nums [0], then the minimum value is on the right, otherwise it is on the left.

[ The mid is greater than the initial values ](https://leetcode.com/problems/search-in-rotated-sorted-array/Figures/33/33_small_mid.png) If the mid is greater than the initial value, then the left side is in order and the minimum value is on the right

[ The mid is smaller than the initial values ](https://leetcode.com/problems/search-in-rotated-sorted-array/Figures/33/33_big_mid.png) If the mid is less than the initial value, then the right side is ordered and the minimum is on the left

```python
# 1. 标准做法
def search(self, nums: List[int], target: int) -> int:
    start, end = 0, len(nums) - 1
    while start <= end:
        mid = start + (end - start) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] >= nums[start]:
            if target >= nums[start] and target < nums[mid]:
                end = mid - 1
            else:
                start = mid + 1
        else:
            if target <= nums[end] and target > nums[mid]:
                start = mid + 1
            else:
                end = mid - 1
    return -1

# 2. 不死循环做法
def findMin(self, nums: List[int]) -> int:
    if nums == None and len(nums) == 0:
        return -1

    start, end = 0, len(nums) - 1

    while start + 1 < end:
        if nums[start] < nums[end]:
            return nums[start]

        mid = start + (end - start) // 2

        # 因为while条件已经限制了3个元素，所以不用担心mid的边界问题
        if nums[mid] > nums[mid - 1]  and nums[mid] > nums[mid + 1]:
            return nums[mid + 1]
        else:
            if nums[mid] > nums[0]:
                start = mid
            else:
                end = mid

    # 这里需要注意，因为我们是找最小值，所以要比较一下start和end的值
    return min(nums[start], nums[end], nums[0])

# 3. bisect 包
import bisect
def search(self, nums: List[int], target: int) -> int:

    if len(nums) == 1:
        return 0 if nums[0] == target else -1

    for i in range(1, len(nums)):
        if nums[i-1] > nums[i]:
            break

    b_po = i
    n = -1

    if nums[0] <= target <= nums[b_po-1]:
        n = bisect.bisect_left(nums[:b_po], target)
    else:
        n = bisect.bisect_left(nums[b_po:], target)
        n = n + b_po

    if n != len(nums)  and nums[n] == target:
        return n
    return -1
```

#### [154. Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)

Test case:

```text
Input: nums = [1,3,5]
Output: 1

Input: nums = [2,2,2,0,1]
Output: 0
```

The only difference between this question and the previous one is that there may be duplicate elements in the array in this question, so we need to do a special treatment, that is, if nums [mid] = nums [start], then we need start + = 1. Because we are not sure which side the minimum value is at this time, we need start + = 1, and then go to the next loop.

```python
def findMin(self, nums: List[int]) -> int:
    if nums == None and len(nums) == 0:
        return -1

    start, end = 0, len(nums) - 1

    while start + 1 < end:
        if nums[start] < nums[end]:
            return nums[start]

        mid = start + (end - start) // 2

        if nums[mid] >= nums[mid - 1] and nums[mid] > nums[mid + 1]:
            return nums[mid + 1]
        else:
            if nums[mid] > nums[0]:
                start = mid
            elif nums[mid] == nums[0]: #
                start += 1
            else:
                end = mid

    return min(nums[start], nums[end], nums[0])
```

#### [278. First Bad Version](https://leetcode.com/problems/first-bad-version/)

Test case:

```text
Input: n = 5, bad = 4
Output: 4
Explanation:
call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true
Then 4 is the first bad version.
```

This question, look carefully, there is no idea, in fact, it is a simple dichotomy, but the dichotomy condition of this question is that if mid is bad version, then end = mid, if mid is not bad version, then start = mid. Finally, you can return either start or end.

```python
def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        start, end = 0, n

        while start + 1 < end:
            mid = start + (end - start) // 2
            if isBadVersion(mid):
                end = mid
            else:
                start = mid

        return start if isBadVersion(start) else end
```

### The answer is two points

#### [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)

Test case:

```text
Input: piles = [3,6,7,11], h = 8
Output: 4

Input: piles = [30,11,23,4,20], h = 5
Output: 30

Input: piles = [30,11,23,4,20], h = 6
Output: 23
```

This problem is actually a variation of the dichotomy. We need to find a minimum speed so that Koko can eat all the bananas in H hours. Then we can use the dichotomy to find the minimum speed.

```python
def minEatingSpeed(self, piles: List[int], h: int) -> int:

    def isValidSpeed(piles, speed, h):
        total = 0
        for item in piles:
            total += (item + speed - 1) // speed
            if total > h:
                return False
        return True

    if len(piles) > h:
        return -1

    start, end = 1, 1000000000

    while start + 1 < end:
        mid = start + (end - start) // 2
        if isValidSpeed(piles, mid, h):
            end = mid
        else:
            start = mid

    if isValidSpeed(piles, start, h):
        return start

    if isValidSpeed(piles, end, h):
        return end

    return -1

```

#### [1283. Find the Smallest Divisor Given a Threshold](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/)

Test case:

```text

Input: nums = [1,2,5,9], threshold = 6
Output: 5
Explanation: We can get a sum to 17 (1+2+5+9) if the divisor is 1.
If the divisor is 4 we can get a sum to 7 (1+1+2+3) and if the divisor is 5 the sum will be 5 (1+1+1+2).
```

This problem is actually a variation of the dichotomy. We need to find a minimum divisor such that sum ( (item + divisor-1)//divisor for item in nums) < = threshold. Then we can use the dichotomy. To find the minimum divisor.

```python
def smallestDivisor(self, nums: List[int], threshold: int) -> int:
    start, end = 1, 1000000

    while start + 1 < end:
        output = 0
        mid = start + (end - start) // 2

        if sum((item + mid - 1) // mid for item in nums) <= threshold:
            end = mid
        else:
            start = mid

    if sum((item + start - 1) // start for item in nums) <= threshold:
        return start

    if sum((item + end - 1) // end for item in nums) <= threshold:
        return end

    return start
```
