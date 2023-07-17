### Foreword

In this topic, we will focus on the specific use of hashmaps.

In many cases, we use hashmaps to store some intermediate results to avoid double counting. This reduces the time complexity from exponential to linear.

In python, we can use `defaultdict` to implement hashmap, which uses the same as `dict`, except that it returns a default value when the key does not exist.

```python
from collections import defaultdict

d = defaultdict(int)

d['a'] += 1

print(d['a']) # 1

print(d['b']) # 0
```

Next, let's look at some specific topics.

### Sum series

#### [1. Two Sum](https://leetcode.com/problems/two-sum/)

The description of this problem is given an array and a target value, find the sum of the two numbers in the array equal to the target value, and return the subscripts of the two numbers.

test cases:

```text
Input: nums = [2,7,11,15], target = 9
Output: [0,1]

Input: nums = [3,2,4], target = 6
Output: [1,2]

Input: nums = [3,3], target = 6
Output: [0,1]
```

The idea of this problem is that we use a hashmap to store the subscript of each number in the array, and then we go through the array, and for each number, we go to the hashmap to see if there is a value for target-num, and if there is, we find both numbers. Otherwise, the current number is stored in the hashmap.

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = defaultdict(int)
        for i, num in enumerate(nums):
            if target - num in d:
                return [d[target - num], i]
            d[num] = i # 这里要注意，我们要先判断再存入，否则会出现重复的情况
```

Complexity analysis:

- Time complexity: O (n). The reason is that we only need to traverse the array once.
- Space complexity: O (n), because we need to use a hashmap to store each number in the array.

#### [219. Contains Duplicate II](https://leetcode.com/problems/contains-duplicate-ii/)

Given an array and an integer K, the description of this problem is to determine whether there are two different subscripts I and J in the array, such that nums [I] = nums [J], and the absolute value of the difference between I and J is at most K.

test cases:

```text
Input: nums = [1,2,3,1], k = 3
Output: true

Input: nums = [1,0,1,1], k = 1
Output: true

Input: nums = [1,2,3,1,2,3], k = 2
Output: false
```

The idea of this problem is that we use a hashmap to store the subscripts of each number in the array, and then we go through the array, and for each number, we go to the hashmap to find out if there is the same number, and the difference between the subscripts is not greater than K, if there is, then we find the two numbers. Otherwise, the current number is stored in the hashmap.

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        d = defaultdict(int)
        for i, num in enumerate(nums):
            if num in d and i - d[num] <= k:
                return True
            d[num] = i
        return False
```

Complexity analysis:

- Time complexity: O (n). The reason is that we only need to traverse the array once.
- Space complexity: O (n), because we need to use a hashmap to store each number in the array.

#### [15. 3Sum](https://leetcode.com/problems/3sum/)

The description of this problem is given an array, find all the triples in the array whose sum is 0, and return these triples.

test cases:

```text
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Input: nums = []
Output: []

Input: nums = [0]
Output: []
```

The idea of this problem is that we first sort the array, then traverse the array, and for each number, we look for two numbers in the array whose sum is 0-num, so that we can convert the sum of three numbers into the sum of two numbers.

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        for i, num in enumerate(nums):
            if i > 0 and nums[i] == nums[i - 1]: # 这里要注意，我们要去重
                continue
            d = defaultdict(int)
            for j in range(i + 1, len(nums)):
                if -num - nums[j] in d:
                    res.append([num, nums[j], -num - nums[j]])
                    while j + 1 < len(nums) and nums[j] == nums[j + 1]: # 这里要注意，我们要去重
                        j += 1
                d[nums[j]] = j
        return res
```

Or we can use the partition method, fix one number at a time, and then use the two sum method to find the other two numbers.

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
                if nums[left] + nums[gap] + nums[right] < 0:
                    left += 1
                elif nums[left] + nums[gap] + nums[right] > 0:
                    right -= 1
                else:
                    res.add((nums[left], nums[gap], nums[right]))
                    left += 1
                    right -= 1
        return res
```

Complexity analysis:

- Time complexity: O (n ^ 2), because we need to traverse the array twice.
- Space complexity: O (n), because we need to use a hashmap to store each number in the array.

#### [18. 4Sum](https://leetcode.com/problems/4sum/)

The description of this problem is given an array and a target value, find all the quadruples in the array whose sum is the target value, and return these quadruples.

test cases:

```text
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

Input: nums = [], target = 0
Output: []

Input: nums = [0], target = 0
Output: []
```

The idea of this problem is that we first sort the array, then traverse the array, and for each number, we look for the three numbers in the array whose sum is target-num, so that we can convert the sum of four numbers into the sum of three numbers.

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        res = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, len(nums)):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                x = target - nums[i] - nums[j]
                start, end = j + 1, len(nums) - 1
                while start < end:
                    if nums[start] + nums[end] == x:
                        res.append([nums[i], nums[j], nums[start], nums[end]])
                        start += 1
                        while start < end and nums[start] == nums[start - 1]:
                            start += 1
                    elif nums[start] + nums[end] < x:
                        start += 1
                    else:
                        end -= 1
        return res
```

Complexity analysis:

- Time complexity: O (n ^ 3), because we need to traverse the array three times.
- Space complexity: O (n), because we need to use a hashmap to store each number in the array.

### Anagram series

#### [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)

The description of this problem is to give two strings and determine whether the two strings are anagram.

test cases:

```text
Input: s = "anagram", t = "nagaram"
Output: true

Input: s = "rat", t = "car"
Output: false
```

The idea of this problem is that we can use a hashmap to store the number of times each character appears in the first string, and then traverse the second string, each time a character is traversed, the number of times the corresponding character in the hashmap is subtracted by one, if the number of characters in the hashmap is less than 0, False, otherwise True.

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        d = defaultdict(int)
        for c in s:
            d[c] += 1
        for c in t:
            d[c] -= 1
            if d[c] < 0:
                return False
        return True
```

Or just use Counter.

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)
```

Complexity analysis:

- Time complexity: O (n). The reason is that we need to traverse the string twice.
- Space complexity: O (n). The reason is that we need to use a hashmap to store the number of occurrences of each character in the first string.

#### [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)

The description of this problem is to give an array of strings and combine all the anagrams in it.

test cases:

```text
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Input: strs = [""]
Output: [[""]]

Input: strs = ["a"]
Output: [["a"]]
```

The idea is that we can use a hashmap to store the string array corresponding to each anagram, and then return all the string arrays in the hashmap.

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = defaultdict(list)
        for s in strs:
            d[''.join(sorted(s))].append(s)
        return list(d.values())
```

Complexity analysis:

- Time complexity: O (NK log K). The reason is that we need to go through the string array once and then sort each string.
- Space complexity: O (NK), because we need to use a hashmap to store the string array corresponding to each anagram.

#### [205. Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/)

The description of this problem is to give two strings and determine whether the two strings are isomorphic.

test cases:

```text
Input: s = "egg", t = "add"
Output: true

Input: s = "foo", t = "bar"
Output: false

Input: s = "paper", t = "title"
Output: true
```

The idea of this problem is that we can use two hashmaps to store the characters in the other string that correspond to each character in the two strings, and then traverse the two strings. If the characters in the two strings correspond to different characters in both hashmaps, then it returns False, otherwise it returns True.

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        s_map, t_map = {}, {}
        for c1, c2 in zip(s, t):
            if (c1 not in s_map) and (c2 not in t_map):
                s_map[c1] = c2
                t_map[c2] = c1

            elif s_map.get(c1) != c2 or t_map.get(c2) != c1:
                return False

        return True
```

Complexity analysis:

- Time complexity: O (n). The reason is that we need to traverse the string twice.
- Space complexity: O (n). The reason is that we need to use two hashmaps to store the characters in one string for each character in the other string.

#### [290. Word Pattern](https://leetcode.com/problems/word-pattern/)

The description of this problem is given a string and a pattern string to determine whether the string conforms to the pattern string.

test cases:

```text
Input: pattern = "abba", s = "dog cat cat dog"
Output: true

Input: pattern = "abba", s = "dog cat cat fish"
Output: false
```

The idea of this problem is that we can use two hashmaps to store the pattern string and each character in the string corresponds to the character in the other string, and then traverse the pattern string and the string, if the characters in the two strings correspond to different characters in the two hashmaps, then it returns False. True otherwise.

```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        s_map, p_map = {}, {}
        s_list = s.split()
        if len(s_list) != len(pattern):
            return False
        for c1, c2 in zip(s_list, pattern):
            if (c1 not in s_map) and (c2 not in p_map):
                s_map[c1] = c2
                p_map[c2] = c1

            elif s_map.get(c1) != c2 or p_map.get(c2) != c1:
                return False

        return True
```

Complexity analysis:

- Time complexity: O (n). The reason is that we need to traverse the string twice.
- Space complexity: O (n). The reason is that we need to use two hashmaps to store the characters in one string for each character in the other string.

#### [202. Happy Number](https://leetcode.com/problems/happy-number/)

The description of this problem is to give an integer and determine whether the integer is happy number.

test cases:

```text
Input: 19
Output: true
Explanation:
1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1
```

The idea of this problem is that we can use a hashmap to store the corresponding sum of squares of each number, and then iterate through each number. If the sum of squares of a number already exists in the hashmap, then it returns False, otherwise it stores the sum of square of the number in the hashmap.

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        visited = set()
        while n != 1:
            if n in visited:
                return False
            visited.add(n)
            n = sum([int(i) ** 2 for i in str(n)])
        return True
```

For the sum line, if it's hard to think of, you can do this

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        visited = set()

        while n != 1 and n not in visited:
            visited.add(n)
            happy_sum = 0

            while n > 0:
                cur = n % 10
                happy_sum += cur * cur
                n = n // 10

            n = happy_sum

        return n == 1
```

Complexity analysis:

- Time complexity: O (n). The reason is that we need to iterate over each number.
- Space complexity: O (n), because we need to use a hashmap to store the sum of squares for each number.

#### [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

The description of this problem is to find the length of the longest consecutive sequence given an unordered array of integers.

test cases:

```text
Input: nums = [100,4,200,1,3,2]
Output: 4

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
```

The idea of this problem is that we can use a hashmap to store the length of the continuous sequence corresponding to each number, and then traverse the array. For each number, we will find out whether there are continuous numbers on the left and right sides of it. If there are, then we will update the length of the continuous sequence of this number to the length plus one on the left and right sides. And then the lengths of the consecutive sequences on the left and right sides are updated.

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums_set = set(nums)
        res = 0
        for num in nums:
            if num - 1 not in nums_set:
                cur = num
                cur_len = 1
                while cur + 1 in nums_set:
                    cur += 1
                    cur_len += 1
                res = max(res, cur_len)
        return res
```

Complexity analysis:

- Time complexity: O (n). The reason is that we need to iterate over each number.
- Space complexity: O (n). The reason is that we need to use a hashmap to store the length of the continuous sequence corresponding to each number.

### Advanced questions

#### [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

The description of this problem is to find the length of the longest consecutive sequence given an unordered array of integers.

test cases:

```text
Input: nums = [100,4,200,1,3,2]
Output: 4

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
```

The idea of this problem is that we can use a hashmap to store the length of the continuous sequence corresponding to each number, and then traverse the array. For each number, we will find out whether there are continuous numbers on the left and right sides of it. If there are, then we will update the length of the continuous sequence of this number to the length plus one on the left and right sides. And then the lengths of the consecutive sequences on the left and right sides are updated.

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums_set = set(nums)
        max_count = 0

        for num in nums_set:
            if num - 1 not in nums_set:
                count = 0
                cur = num

                while cur in nums_set:
                    cur = cur + 1
                    count += 1

                max_count = max(count, max_count)

        return max_count
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the array once.
- Space complexity: O (n). The reason is that we need to use a hashmap to store the length of the continuous sequence corresponding to each number.
