704. Binary Search

```python
def search(self, nums: List[int], target: int) -> int:
    if nums == None or len(nums) == 0:
        return -1
    
    start, end = 0, len(nums) - 1
    
    while start + 1 < end:
        mid = start + (end - start) // 2
        if nums[mid] > target:
            end = mid
        elif nums[mid] < target:
            start = mid
        else:
            return mid
        
    if nums[start] == target:
        return start
    
    if nums[end] == target:
        return end
    
    return -1
```

34.

```python
def searchRange(self, nums: List[int], target: int) -> List[int]:
    def findLeft(nums, target):
        start, end = 0, len(nums) - 1
    
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] >= target:
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
            else:
                start = mid

        if nums[end] == target:
            return end
        if nums[start] == target:
            return start
        return -1

    if nums == None or len(nums) == 0:
        return [-1, -1]

    
    return [findLeft(nums, target), findRight(nums,target)]

```


702.

```python
def search(self, reader, target):
    start, end = 0, 1
    while reader.get(end) < target:
        end <<= 2
    
    print(end)
    
    while start + 1 < end:
        mid = start + (end - start) // 2
        if reader.get(mid) > target:
            end = mid
        elif reader.get(mid) < target:
            start = mid
        else:
            return mid

    if reader.get(start) == target:
        return start

    if reader.get(end) == target:
        return end

    return -1
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