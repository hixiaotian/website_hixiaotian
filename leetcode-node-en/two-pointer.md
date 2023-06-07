### Preface

In the context of two-pointer problems, there are two types: disparate two pointers and identical two pointers.

Disparate two pointers generally refer to one pointer moving from left to right and another pointer moving from right to left. For example, in binary search, we use disparate two pointers.

Identical two pointers generally refer to both pointers moving from left to right or both moving from right to left. For example, in linked lists, we use identical two pointers.

Next, we will discuss various approaches using two pointers. Let's take a look!

### Basic Problems

<b>↓ Click on the problem title to directly navigate to the corresponding LeetCode page ↓</b>

#### [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)

The description of this problem is to determine whether a string is a palindrome.

Test cases:

```text
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
```

We can solve this problem using two pointers. One pointer starts from the left, and the other pointer starts from the right. We compare the characters pointed to by the two pointers. If they are not equal, then the string is not a palindrome. If they are equal, we continue moving towards the center.

The code is as follows:

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

- Time complexity: O(n), because each character is traversed once.
- Space complexity: O(1), as only a constant number of variables are used.

#### [680. Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)

The description of this problem is to determine whether a string is a palindrome after deleting at most one character.

Test cases:

```text
Input: s = "aba"
Output: true

Input: s = "abca"
Output: true
Explanation: You could delete the character 'c'.

Input: s = "abc"
Output: false
```

We can solve this problem using two pointers. One pointer starts from the left, and the other pointer starts from the right. We compare the characters pointed to by the two pointers. If they are not equal, we have two options: delete the character pointed to by the left pointer or delete the character pointed to by the right pointer. Then, we check if the remaining string is a palindrome. If it is, we return true; otherwise, we return false.

The code is as follows:

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        start, end = 0, len(s) - 1

        while start < end:
            if s[start] != s[end]:
                left = s[start: end]
                right = s[start + 1: end + 1]
                return left == left[::-1] or right == right[::-1]

            start += 1
            end -= 1

        return True
```

Complexity analysis:

- Time complexity: O(n), because each character is traversed once.
- Space complexity: O(1), as only a constant number of variables are used.

#### [392. Is Subsequence](https://leetcode.com/problems/is-subsequence/)

The description of this problem is to determine whether a string is a subsequence of another string.

Test cases:

```text
Input: s = "abc", t = "ahbgdc"
Output: true

Input: s = "axc", t = "ahbgdc"
Output: false
```

We can solve this problem using two pointers. One pointer starts from the left, and the other pointer starts from the right. We compare the characters pointed to by the two pointers. If they are equal, we move both pointers to the right. If they are not equal, we only move the pointer on the right. Finally, we check if the pointer on the left has reached the end of the string.

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

- Time complexity: O(n), because each character is traversed once.
- Space complexity: O(1), as only a constant number of variables are used.

#### [167. Two Sum II - Input array is sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

The description of this problem is to find two numbers in an array that add up to a specific target.

Test cases:

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

We can solve this problem using two pointers. One pointer starts from the left, and the other pointer starts from the right. We compare the sum of the two numbers pointed to by the two pointers with the target. If the sum is equal to the target, we return the indices of the two numbers. If the sum is greater than the target, we move the pointer on the right to the left. If the sum is less than the target, we move the pointer on the left to the right.

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

- Time complexity: O(n), because each number is traversed once.
- Space complexity: O(1), as only a constant number of variables are used.

#### [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

The description of this problem is to find two lines that form a container with the most water.

test cases:
![image.png](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

```text
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
```

We can solve this problem using two pointers. One pointer starts from the left, and the other pointer starts from the right. We compare the area of the two lines pointed to by the two pointers. If the area is greater than the maximum area, we update the maximum area. Then, we move the pointer on the left to the right if the line pointed to by the pointer on the left is shorter than the line pointed to by the pointer on the right; otherwise, we move the pointer on the right to the left.

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

- Time complexity: O(n), because each number is traversed once.
- Space complexity: O(1), as only a constant number of variables are used.

#### [15. 3Sum](https://leetcode.com/problems/3sum/)

The description of this problem is to find all unique triplets in the array which gives the sum of zero.

Test cases:

```text
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Input: nums = []
Output: []

Input: nums = [0]
Output: []
```

We can solve this problem using two pointers. One pointer starts from the left, and the other pointer starts from the right. We compare the sum of the three numbers pointed to by the two pointers with zero. If the sum is equal to zero, we add the three numbers to the result. If the sum is greater than zero, we move the pointer on the right to the left. If the sum is less than zero, we move the pointer on the left to the right.

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

We can also use "gap" as a board to seperate the array into two parts. The left part is the numbers that are less than the gap, and the right part is the numbers that are greater than the gap. We can use two pointers to find the two numbers that add up to the gap. If the sum of the two numbers is equal to the gap, we add the three numbers to the result. If the sum is greater than the gap, we move the pointer on the right to the left. If the sum is less than the gap, we move the pointer on the left to the right.

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
                if nums[left] + nums[gap] + nums[right] < 0:  # move left pointer
                    left += 1
                elif nums[left] + nums[gap] + nums[right] > 0: # move right pointer
                    right -= 1
                else:
                    res.add((nums[left], nums[gap], nums[right]))
                    left += 1
                    right -= 1
        return res
```

Complexity analysis:

- Time complexity: O(n^2), because each number is traversed once.
- Space complexity: O(n), as a set is used to store the result.
