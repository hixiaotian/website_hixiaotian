Binary Search

### 前言
在二分法中，需要熟练掌握模版，这里提供两个模版，一个是标准模版，另一个是不死模版。
标准模版是大多数leetcode模版里提供的答案，而不死模版是来自九章算法的“永不死循环模版”，各有各的好处。

当然，python也提供bisect包，不过在面试中，非常不建议使用，不过对于OA解题，很推荐使用这个，因为省时省力。

接下来以在一个有序数组中搜寻一个target为例题，比较这仨者的区别。
<br>
#### [704. Binary Search](https://leetcode.com/problems/binary-search/)
test cases:
```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
```

标准模版（注意事项已标注在代码中）：
![二分法基本图解](https://assets.leetcode.com/static_assets/posts/1EYkSkQaoduFBhpCVx7nyEA.gif)

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

而在不死循环模版里，我们把while的条件稍作修改（注意事项已标注在代码中）：
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

直接使用bisect包，方便省事，但不推荐面试中使用：
```python
from bisect import bisect_left
def search(self, nums: List[int], target: int) -> int:
    num_size = len(nums)
    i = bisect_left(nums, target)        
    return i if i < num_size and nums[i] == target else -1
```

二分法解法的复杂度分析：  
Time complexity: O(log⁡N)
Space complexity: O(1)


#### [35. Search Insert Position](https://leetcode.com/problems/search-insert-position)
这道题与前面那道题不同的一点，就是他要决定插入地点，那么这个就只需要修改一下最后条件即可。

test cases:
```
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

解法的复杂度分析：  
Time complexity: O(log⁡N)
Space complexity: O(1)


#### [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

这道题涉及重复元素，那么要注意，在哪里停下呢？对if条件做修改即可
test cases:
```
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

解法的复杂度分析：  
Time complexity: O(log⁡N)
Space complexity: O(1)


#### [702. Search in a Sorted Array of Unknown Size] (https://leetcode.com/problems/search-in-a-sorted-array-of-unknown-size/description/)

Test cases:
```
Input: secret = [-1,0,3,5,9,12, ...], target = 9
Output: 4
Explanation: 9 exists in secret and its index is 4.

Input: secret = [-1,0,3,5,9,12, ...], target = 2
Output: -1
Explanation: 2 does not exist in secret so return -1.

```

这个基本思路跟前面的一样，只有一个判断end点在哪，那么我们只需要加一个地方，二分的解法与前面雷同啦！！
```python
def search(self, reader, target):
    start, end = 0, 1
    # 比target小，就疯狂让end 以次方级别增加
    while reader.get(end) < target:
        end <<= 1
    
    # 这里用之前的二分解法就可以了，思路不变
    ...
```



153. Find Minimum in Rotated Sorted Array

```python
def findMin(self, nums: List[int]) -> int:
    if nums == None and len(nums) == 0:
        return -1
    
    start, end = 0, len(nums) - 1
    
    while start + 1 < end:
        if nums[start] < nums[end]:
            return nums[start]
        
        mid = start + (end - start) // 2 
    
        if nums[mid] > nums[mid - 1]  and nums[mid] > nums[mid + 1]:
            return nums[mid + 1]
        else:
            if nums[mid] > nums[0]:
                start = mid
            else:
                end = mid
                
    return min(nums[start], nums[end], nums[0])
```

154. Find Minimum in Rotated Sorted Array II
```python
def findMin(self, nums: List[int]) -> int:
        if nums == None and len(nums) == 0:
            return -1

        start, end = 0, len(nums) - 1

        while start + 1 < end:
            if nums[start] < nums[end]:
                return nums[start]

            mid = start + (end - start) // 2 

            if nums[mid] >= nums[mid - 1]  and nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            else:
                if nums[mid] > nums[0]:
                    start = mid
                elif nums[mid] == nums[0]:
                    start += 1
                else:
                    end = mid

        return min(nums[start], nums[end], nums[0])
```


278. First Bad Version
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


875. Koko Eating bananas

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

1283. Find the Smallest Divisor Given a Threshold
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