### Foreword

In the double pointer problem, there are two cases, the first is the opposite double pointer, the second is the same double pointer.

The opposite direction double pointer generally means that one pointer is from left to right and the other pointer is from right to left. For example, in binary search, we use the opposite direction double pointer.

Double pointers in the same direction generally mean that both pointers are from left to right or from right to left. For example, in a linked list, we use double pointers in the same direction.

Next, there are some questions to discuss the use of various double pointers. Let's take a look first!

### Basic questions

<b>↓ Click on the title to jump directly to the leetcode title page ↓</b>

#### [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)

The description of this problem is to determine whether a string is a palindrome string.

test cases:

```text
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
```

We can do it with double pointers, one pointer from left to right, one pointer from right to left, and then judge whether the characters pointed by the two pointers are equal, if not, then it is not a palindrome string, if it is equal, then it continues to move to the middle.

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        left, right = 0, len(s) - 1
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        return True
```

Complexity analysis:

- Time complexity: O (n), because each character is traversed once
- Space complexity: O (1), because only a constant number of variables are used

#### [680. Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)

This question is a variation of the previous question. The description of this question is that given a string, we can delete a character to determine whether it can form a palindrome string.

test cases:

```text
Input: s = "aba"
Output: true

Input: s = "abca"
Output: true
Explanation: You could delete the character 'c'.

Input: s = "abc"
Output: false
```

We can do this with two pointers, one pointer from left to right, one pointer from right to left, and then determine whether the characters pointed to by the two pointers are equal, if not, then delete the character pointed to by the left pointer, or delete the character pointed by the right pointer, and then determine whether the remaining string is a palindrome string, if so, then return true. Otherwise, it returns false.

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        start, end = 0, len(s) - 1

        while start < end:
            if s[start] != s[end]:
                left = s[start: end] # 删除左指针指向的字符
                right = s[start + 1: end + 1] # 删除右指针指向的字符
                return left == left[::-1] or right == right[::-1]

            start += 1
            end -= 1

        return True
```

Complexity analysis:

- Time complexity: O (n), because each character is traversed once
- Space complexity: O (1), because only a constant number of variables are used

#### [392. Is Subsequence](https://leetcode.com/problems/is-subsequence/)

The description of this problem is, given two strings s and t, to determine whether s is a subsequence of t.

test cases:

```text
Input: s = "abc", t = "ahbgdc"
Output: true

Input: s = "axc", t = "ahbgdc"
Output: false
```

We can do this with two pointers in the same direction, one pointer pointing to s, one pointer pointing to t, and then determine whether the characters pointed by the two pointers are equal, if they are equal, then continue to move backward, if not, then continue to move backward the t pointer until the character pointed by the t pointer is equal to the character pointed by the s pointer. And then continue to move back.

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == len(s)
```

Complexity analysis:

- Time complexity: O (n), because each character is traversed once
- Space complexity: O (1), because only a constant number of variables are used

#### [167. Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

The description of this problem is that given an ascending array and a target value, find the sum of the two numbers in the array equal to the target value, and return the subscripts of the two numbers.

test cases:

```text
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].

Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 4 is 6. Therefore index1 = 1, index2 = 3. We return [1, 3].

Input: numbers = [-1,0], target = -1
Output: [1,2]
Explanation: The sum of -1 and 0 is -1. Therefore index1 = 1, index2 = 2. We return [1, 2].
```

We can do this with a negative double pointer, one pointer to the leftmost side of the array, one pointer to the rightmost side of the array, and then determine whether the sum of the numbers pointed to by the two pointers is equal to the target value. If it is equal, then return the subscripts of the two pointers. If it is less than the target value, then move the left pointer right, and if it is greater than the target value, then move the right pointer left.

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        while left < right:
            if numbers[left] + numbers[right] == target:
                return [left + 1, right + 1]
            elif numbers[left] + numbers[right] < target:
                left += 1
            else:
                right -= 1
        return [-1, -1]
```

Complexity analysis:

- Time complexity: O (n), because each character is traversed once
- Space complexity: O (1), because only a constant number of variables are used

#### [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

The description of this problem is that given an array, each element in the array represents the height of a column, and then find two columns such that the area enclosed by the two columns and the x-axis is the largest.

test cases:![image.png](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

```text
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
```

We can do it with a negative double pointer, one pointer points to the leftmost side of the array, one pointer points to the rightmost side of the array, and then calculate the area enclosed by the columns pointed to by the two pointers, and then judge the height of the columns pointed to by the two pointers, if the height of the column pointed by the left pointer is less than that pointed by the right pointer, then move the left pointer to the right. If the height of the column to which the left pointer points is greater than the height of the column to which the right pointer points, the right pointer is moved to the left.

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        res = 0
        while left < right:
            res = max(res, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return res
```

Complexity analysis:

- Time complexity: O (n), because each character is traversed once
- Space complexity: O (1), because only a constant number of variables are used

#### [15. 3Sum](https://leetcode.com/problems/3sum/)

The description of this problem is that given an array, find the triples of all three numbers in the array whose sum is equal to 0, and return these triples.

test cases:

```text
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Input: nums = []
Output: []

Input: nums = [0]
Output: []
```

We can do this with a negative double pointer, one pointer to the leftmost side of the array, one pointer to the rightmost side of the array, and then determine whether the sum of the numbers pointed to by the two pointers is equal to the target value. If it is equal, then return the subscripts of the two pointers. If it is less than the target value, then move the left pointer right, and if it is greater than the target value, then move the right pointer left.

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        for i in range(len(nums)):
            if nums[i] > 0:
                break
            if i == 0 or nums[i - 1] != nums[i]:
                self.twoSum(nums, i, res)
        return res

    def twoSum(self, nums: List[int], i: int, res: List[List[int]]):
        seen = set()
        j = i + 1
        while j < len(nums):
            complement = -nums[i] - nums[j]
            if complement in seen:
                res.append([nums[i], nums[j], complement])
                while j + 1 < len(nums) and nums[j] == nums[j + 1]:
                    j += 1
            seen.add(nums[j])
            j += 1
```

We can also use the partition method to do this, gap as a partition, the left pointer points to the left of the partition, the right pointer points to the right of the partition, and then determine whether the number pointed to by the left pointer plus the number pointed to by the partition plus the number pointed by the right pointer is equal to 0, if equal to 0, then return the subscripts of the three pointers, if less than 0, then move the left pointer to the right. If it is greater than 0, the right pointer is moved to the left.

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        if not nums or len(nums) < 3:
            return []

        res = set()
        for gap in range(1, len(nums) - 1):
            left = 0
            right = len(nums) - 1
            while left < gap and right > gap:
                if nums[left] + nums[gap] + nums[right] < 0: # 说明左边的数太小了，需要增大
                    left += 1
                elif nums[left] + nums[gap] + nums[right] > 0: # 说明右边的数太大了，需要减小
                    right -= 1
                else:
                    res.add((nums[left], nums[gap], nums[right]))
                    left += 1
                    right -= 1
        return res
```

Complexity analysis:

- Time complexity: O (n ^ 2), because each character is traversed once
- Space complexity: O (1), because only a constant number of variables are used

### In place substitution problem

#### [283. Move Zeroes](https://leetcode.com/problems/move-zeroes/)

The description of the problem is that given an array, move the 0 in the array to the end of the array while maintaining the relative order of the non-zero elements.

test cases:

```text
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```

The idea of this problem is to use two pointers fast and slow, the fast pointer is used to traverse the array, the slow pointer is used to point to the position of the non-zero element, then the element pointed by the fast pointer is assigned to the element pointed by the slow pointer, and then the fast pointer moves forward one step. The slow pointer moves forward by one step until the element pointed by the fast pointer is 0, then the fast pointer moves forward by one step, the slow pointer does not move until the element pointed by the fast pointer is not 0, and then the element pointing to the slow pointer is assigned with a value, Then the fast pointer moves one step forward and the slow pointer moves one step forward until the fast pointer reaches the end of the array, and then the element pointed to by the slow pointer is assigned a value of 0 to the end of the array.

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        slow, fast = 0, 0
        while fast < len(nums):
            while fast + 1 < len(nums) and nums[fast] == 0:
                fast += 1
            nums[slow] = nums[fast]

            slow += 1
            fast += 1


        while slow < len(nums):
            if nums[slow] != 0:
                nums[slow] = 0
            slow += 1
```

Complexity analysis:

- Time complexity: O (n), because each character is traversed once
- Space complexity: O (1), because only a constant number of variables are used

#### [27. Remove Element](https://leetcode.com/problems/remove-element/)

The description of this problem is that given an array and a value, all the elements in the array equal to this value are deleted, and the length of the array after deletion is returned.

test cases:

```text
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2]
```

The idea of this problem is to use two pointers fast and slow, the fast pointer is used to traverse the array, the slow pointer is used to point to the position of the non-val element, and then assign the element pointed by the fast pointer to the element pointed by the slow pointer, and then the fast pointer moves forward one step. The slow pointer moves forward by one step until the element pointed by the fast pointer is Val, then the fast pointer moves forward by one step, the slow pointer does not move until the element pointed by the fast pointer is not Val, and then the element pointed to by the slow pointer is assigned, The fast pointer then moves forward one step and the slow pointer moves forward one step until the fast pointer reaches the end of the array and then returns the value of the slow pointer.

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        slow, fast = 0, 0
        while fast < len(nums):
            while fast + 1 < len(nums) and nums[fast] == val:
                fast += 1
            nums[slow] = nums[fast]

            slow += 1
            fast += 1

        # 如果最后一个元素是 val，那么 slow 指针指向的元素就是 val，所以需要减 1
        if len(nums) >= 1 and nums[-1] == val:
            return slow - 1

        return slow
```

Complexity analysis:

- Time complexity: O (n), because each character is traversed once
- Space complexity: O (1), because only a constant number of variables are used

#### [26. Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

The description of this problem is that given a sorted array, delete duplicate elements so that each element appears only once, and return the length of the array after deletion.

test cases:

```text
Input: nums = [1,1,2]
Output: 2, nums = [1,2]
```

The idea of this problem is to use two pointers, fast and slow, the fast pointer is used to traverse the array, the slow pointer is used to point to the position of the non-repeating element, and then assign the element pointed by the fast pointer to the element pointed by the slow pointer, and then the fast pointer moves forward one step. The slow pointer moves forward one step until the element pointed to by the fast pointer and the element pointed to by the slow pointer are not equal, then the element pointed by the fast pointer is assigned to the element pointed by the slow pointer, and then the fast pointer moves forward one step, and the slow pointer moves forwards one step, Until the fast pointer reaches the end of the array, and then returns the value of the slow pointer.

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        slow, fast = 0, 0
        while fast < len(nums):
            while fast + 1 < len(nums) and nums[fast] == nums[fast + 1]:
                fast += 1
            nums[slow] = nums[fast]

            slow += 1
            fast += 1

        return slow
```

Complexity analysis:

- Time complexity: O (n), because each character is traversed once
- Space complexity: O (1), because only a constant number of variables are used

#### [80. Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)

The description of this problem is that given a sorted array, delete duplicate elements so that each element appears at most twice, and return the length of the array after deletion.

test cases:

```text
Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3]
```

The difference between this question and the previous one is that each element appears only once in the previous question, while in this question, each element appears only twice at most, so we need to add a variable count to record the number of times the current element appears. When count is greater than or equal to 2, The element pointed by the fast pointer is no longer assigned to the element pointed by the slow pointer, but the fast pointer is only moved until the element pointed to by the fast pointer and the element pointed to by the slow pointer are not equal to each other, and then the element pointing to the fast needle is assigned to the element pointing to the slow needle, The fast pointer then moves forward one step and the slow pointer moves forward one step until the fast pointer reaches the end of the array and then returns the value of the slow pointer.

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        slow, fast = 0, 0
        count = 0
        while fast < len(nums):
            while fast + 1 < len(nums) and nums[fast] == nums[fast + 1]:
                fast += 1
                count += 1
                if count < 2:
                    nums[slow] = nums[fast]
                    slow += 1
            nums[slow] = nums[fast]
            slow += 1
            fast += 1
            count = 0

        return slow
```
