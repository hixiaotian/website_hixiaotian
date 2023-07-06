### 前言

在这个专题，我们要重点讨论 hashmap 的具体用法。

在很多时候，我们使用 hashmap 来存储一些中间结果，以避免重复计算。这样可以使得时间复杂度从指数级降低到线性级。

在 python 当中，我们可以使用`defaultdict`来实现 hashmap，它的用法和`dict`是一样的，只不过它可以在 key 不存在的时候返回一个默认值。

```python
from collections import defaultdict

d = defaultdict(int)

d['a'] += 1

print(d['a']) # 1

print(d['b']) # 0
```

接下来我们就来看一些具体的题目。

### Sum 系列

#### [1. Two Sum](https://leetcode.com/problems/two-sum/)

这道题的描述是给定一个数组和一个目标值，找到数组中两个数的和等于目标值，并返回这两个数的下标。

test cases:

```text
Input: nums = [2,7,11,15], target = 9
Output: [0,1]

Input: nums = [3,2,4], target = 6
Output: [1,2]

Input: nums = [3,3], target = 6
Output: [0,1]
```

这道题的思路是，我们使用一个 hashmap 来存储数组中每个数的下标，然后遍历数组，对于每个数，我们都去 hashmap 中查找是否存在 target - num 的值，如果存在，那么就找到了这两个数，否则就将当前的数存入 hashmap 中。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = defaultdict(int)
        for i, num in enumerate(nums):
            if target - num in d:
                return [d[target - num], i]
            d[num] = i # 这里要注意，我们要先判断再存入，否则会出现重复的情况
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们只需要遍历一遍数组。
- 空间复杂度：O(n)，原因是我们需要使用一个 hashmap 来存储数组中的每个数。

#### [219. Contains Duplicate II](https://leetcode.com/problems/contains-duplicate-ii/)

这道题的描述是给定一个数组和一个整数 k，判断数组中是否存在两个不同的下标 i 和 j，使得 nums[i] = nums[j]，并且 i 和 j 的差的绝对值最大为 k。

test cases:

```text
Input: nums = [1,2,3,1], k = 3
Output: true

Input: nums = [1,0,1,1], k = 1
Output: true

Input: nums = [1,2,3,1,2,3], k = 2
Output: false
```

这道题的思路是，我们使用一个 hashmap 来存储数组中每个数的下标，然后遍历数组，对于每个数，我们都去 hashmap 中查找是否存在相同的数，并且下标之差不大于 k，如果存在，那么就找到了这两个数，否则就将当前的数存入 hashmap 中。

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

复杂度分析：

- 时间复杂度：O(n)，原因是我们只需要遍历一遍数组。
- 空间复杂度：O(n)，原因是我们需要使用一个 hashmap 来存储数组中的每个数。

#### [15. 3Sum](https://leetcode.com/problems/3sum/)

这道题的描述是给定一个数组，找到数组中所有和为 0 的三元组，并返回这些三元组。

test cases:

```text
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Input: nums = []
Output: []

Input: nums = [0]
Output: []
```

这道题的思路是，我们先对数组进行排序，然后遍历数组，对于每个数，我们都去寻找数组中和为 0-num 的两个数，这样就可以将三数之和转化为两数之和。

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

或者说我们可以使用隔板法，每次固定一个数，然后再使用 two sum 的方法来找到另外两个数。

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

复杂度分析：

- 时间复杂度：O(n^2)，原因是我们需要遍历两遍数组。
- 空间复杂度：O(n)，原因是我们需要使用一个 hashmap 来存储数组中的每个数。

#### [18. 4Sum](https://leetcode.com/problems/4sum/)

这道题的描述是给定一个数组和一个目标值，找到数组中所有和为目标值的四元组，并返回这些四元组。

test cases:

```text
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

Input: nums = [], target = 0
Output: []

Input: nums = [0], target = 0
Output: []
```

这道题的思路是，我们先对数组进行排序，然后遍历数组，对于每个数，我们都去寻找数组中和为 target-num 的三个数，这样就可以将四数之和转化为三数之和。

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

复杂度分析：

- 时间复杂度：O(n^3)，原因是我们需要遍历三遍数组。
- 空间复杂度：O(n)，原因是我们需要使用一个 hashmap 来存储数组中的每个数。

### Anagram 系列

#### [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)

这道题的描述是给定两个字符串，判断这两个字符串是否是 anagram。

test cases:

```text
Input: s = "anagram", t = "nagaram"
Output: true

Input: s = "rat", t = "car"
Output: false
```

这道题的思路是，我们可以使用一个 hashmap 来存储第一个字符串中每个字符出现的次数，然后再遍历第二个字符串，每遍历到一个字符，就将 hashmap 中对应的字符的次数减一，如果 hashmap 中对应的字符的次数小于 0，那么就返回 False，否则返回 True。

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

或者直接使用 Counter。

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历两遍字符串。
- 空间复杂度：O(n)，原因是我们需要使用一个 hashmap 来存储第一个字符串中每个字符出现的次数。

#### [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)

这道题的描述是给定一个字符串数组，将其中所有 anagram 组合在一起。

test cases:

```text
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Input: strs = [""]
Output: [[""]]

Input: strs = ["a"]
Output: [["a"]]
```

这道题的思路是，我们可以使用一个 hashmap 来存储每个 anagram 对应的字符串数组，然后再将 hashmap 中的所有字符串数组返回。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = defaultdict(list)
        for s in strs:
            d[''.join(sorted(s))].append(s)
        return list(d.values())
```

复杂度分析：

- 时间复杂度：O(nklogk)，原因是我们需要遍历一遍字符串数组，然后对每个字符串进行排序。
- 空间复杂度：O(nk)，原因是我们需要使用一个 hashmap 来存储每个 anagram 对应的字符串数组。

#### [205. Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/)

这道题的描述是给定两个字符串，判断这两个字符串是否是同构的。

test cases:

```text
Input: s = "egg", t = "add"
Output: true

Input: s = "foo", t = "bar"
Output: false

Input: s = "paper", t = "title"
Output: true
```

这道题的思路是，我们可以使用两个 hashmap 来存储两个字符串中每个字符对应的另一个字符串中的字符，然后再遍历两个字符串，如果两个字符串中的字符在两个 hashmap 中对应的字符不相同，那么就返回 False，否则返回 True。

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

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历两遍字符串。
- 空间复杂度：O(n)，原因是我们需要使用两个 hashmap 来存储两个字符串中每个字符对应的另一个字符串中的字符。

#### [290. Word Pattern](https://leetcode.com/problems/word-pattern/)

这道题的描述是给定一个字符串和一个模式串，判断这个字符串是否符合这个模式串。

test cases:

```text
Input: pattern = "abba", s = "dog cat cat dog"
Output: true

Input: pattern = "abba", s = "dog cat cat fish"
Output: false
```

这道题的思路是，我们可以使用两个 hashmap 来存储模式串和字符串中每个字符对应的另一个字符串中的字符，然后再遍历模式串和字符串，如果两个字符串中的字符在两个 hashmap 中对应的字符不相同，那么就返回 False，否则返回 True。

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

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历两遍字符串。
- 空间复杂度：O(n)，原因是我们需要使用两个 hashmap 来存储两个字符串中每个字符对应的另一个字符串中的字符。

#### [202. Happy Number](https://leetcode.com/problems/happy-number/)

这道题的描述是给定一个整数，判断这个整数是否是 happy number。

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

这道题的思路是，我们可以使用一个 hashmap 来存储每个数对应的平方和，然后再遍历每个数，如果某个数的平方和在 hashmap 中已经存在了，那么就返回 False，否则就将这个数的平方和存入 hashmap 中。

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

对于 sum 那一行，如果很难想到，可以这样做

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

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历每个数。
- 空间复杂度：O(n)，原因是我们需要使用一个 hashmap 来存储每个数对应的平方和。

#### [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

这道题的描述是给定一个无序的整数数组，找到其中最长的连续序列的长度。

test cases:

```text
Input: nums = [100,4,200,1,3,2]
Output: 4

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
```

这道题的思路是，我们可以使用一个 hashmap 来存储每个数对应的连续序列的长度，然后遍历数组，对于每个数，我们都去寻找它的左右两边是否有连续的数，如果有，那么就将这个数的连续序列的长度更新为左右两边的连续序列的长度加一，然后再更新左右两边的连续序列的长度。

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

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历每个数。
- 空间复杂度：O(n)，原因是我们需要使用一个 hashmap 来存储每个数对应的连续序列的长度。

### 高级题

#### [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

这道题的描述是给定一个无序的整数数组，找到其中最长的连续序列的长度。

test cases:

```text
Input: nums = [100,4,200,1,3,2]
Output: 4

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
```

这道题的思路是，我们可以使用一个 hashmap 来存储每个数对应的连续序列的长度，然后遍历数组，对于每个数，我们都去寻找它的左右两边是否有连续的数，如果有，那么就将这个数的连续序列的长度更新为左右两边的连续序列的长度加一，然后再更新左右两边的连续序列的长度。

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

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历一遍数组。
- 空间复杂度：O(n)，原因是我们需要使用一个 hashmap 来存储每个数对应的连续序列的长度。
