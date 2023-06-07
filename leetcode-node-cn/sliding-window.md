### 前言

在这个专题，我们主要讨论 sliding window 的具体用法。

针对很多类型的问题都可以使用 sliding window，比如说在一个字符串中找到最长的不重复子串，或者是在一个数组中找到最长的连续子数组等等。

我们简单介绍一下 sliding window 的模版：

```python
def slidingWindow(s: str):
    window = defaultdict(int)
    left, right = 0, 0
    while right < len(s):
        cur_right = s[right]
        window[cur_right] += 1
        right += 1
        # 进行窗口内数据的一系列更新
        ...

        # debug 输出的位置
        print(f"window: [{left}, {right})")

        # 判断左侧窗口是否要收缩
        while left < right and window needs shrink:
            cur_left = s[left]
            window[cur_left] -= 1
            if window[cur_left] == 0:
                del window[cur_left]
            left += 1
            # 进行窗口内数据的一系列更新
            ...
```

这个模版的核心思想是：

- 1. 先不断地增加 right 指针扩大窗口
- 2. 当窗口包含了 t 的所有字符后，开始不断地收缩 left 指针，直到得到最小窗口

#### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

这道题的描述是找到最长的不重复子串，我们可以使用 sliding window 来解决。

test cases:

```text
Input: s = "abcabcbb"
Output: 3

Input: s = "bbbbb"
Output: 1

Input: s = "pwwkew"
Output: 3
```

这道题的思路是，我们使用一个 window 来存储当前的子串，然后不断地增加 right 指针来扩大窗口，直到窗口中出现了重复的字符，此时我们就需要收缩 left 指针来缩小窗口，直到窗口中不再有重复的字符。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        window = defaultdict(int)
        left, right = 0, 0
        res = 0

        while right < len(s):
            cur_right = s[right]
            window[cur_right] += 1
            right += 1

            while window[cur_right] > 1:
                cur_left = s[left]
                window[cur_left] -= 1
                if window[cur_left] == 0:
                    del window[cur_left]
                left += 1

            res = max(res, right - left)

        return res
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个字符串。
- 空间复杂度：O(n)，因为我们使用了一个 window 来存储当前的子串。

#### [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

这道题的描述是找到最小的包含 t 的子串，我们可以使用 sliding window 来解决。

test cases:

```text
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"

Input: s = "a", t = "a"
Output: "a"
```

这道题的思路是，我们使用一个 window 来存储当前的子串，然后不断地增加 right 指针来扩大窗口，直到窗口中包含了 t 的所有字符，此时我们就需要收缩 left 指针来缩小窗口，直到窗口中不再包含 t 的所有字符。

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        left, right = 0, 0
        min_start, min_length = None, float("inf")
        window = collections.Counter()
        target = collections.Counter(t)
        valid = 0 # 记录 window 中已经有多少字符符合要求了

        while right < len(s):
            cur_right = s[right]
            window[cur_right] += 1

            if window[cur_right] == target[cur_right]:
                valid += 1 # 如果 window 中 cur_right 字符的数量符合要求了，那么 valid += 1

            right += 1

            while left <= right and valid >= len(target): # 如果 window 中包含了所有字符，那么就开始收缩 left 指针
                if right - left < min_length: # 如果当前的窗口大小更小，则更新 min_length
                    min_start = left
                    min_length = right - left

                cur_left = s[left]
                window[cur_left] -= 1
                if cur_left in target: # 如果 window 中 cur_left 字符的数量不再符合要求了，那么 valid -= 1
                    if window[cur_left] < target[cur_left]:
                        valid -= 1
                left += 1

        return s[min_start: min_start + min_length] if min_length != float("inf") else ""
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个字符串。
- 空间复杂度：O(n)，因为我们使用了一个 window 来存储当前的子串。

#### [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)

这道题的描述是找到最小的连续子数组，使得子数组的和大于等于 s，我们可以使用 sliding window 来解决。

test cases:

```text
Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2

Input: s = 4, nums = [1,4,4]
Output: 1

Input: s = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0
```

这道题的思路是，我们使用一个 window 来存储当前的子串，然后不断地增加 right 指针来扩大窗口，直到窗口中的和大于等于 s，此时我们就需要收缩 left 指针来缩小窗口，直到窗口中的和小于 s。

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        left, right = 0, 0
        res = float("inf")
        window = 0

        while right < len(nums):
            cur_right = nums[right]
            window += cur_right
            right += 1

            while window >= s:
                res = min(res, right - left)
                cur_left = nums[left]
                window -= cur_left
                left += 1

        return res if res != float("inf") else 0
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个数组。
- 空间复杂度：O(1)，因为我们使用了一个 window 来存储当前的子串。

#### [567. Permutation in String](https://leetcode.com/problems/permutation-in-string/)

这道题的描述是判断 s2 中是否包含 s1 的排列，我们可以使用 sliding window 来解决。

test cases:

```text
Input: s1 = "ab", s2 = "eidbaooo"
Output: True

Input: s1= "ab", s2 = "eidboaoo"
Output: False
```

这道题的思路是，我们使用一个 window 来存储当前的子串，然后不断地增加 right 指针来扩大窗口，直到窗口中包含了 s1 的所有字符，此时我们就需要收缩 left 指针来缩小窗口，直到窗口中不再包含 s1 的所有字符。

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        left, right = 0, 0
        window = collections.Counter()
        target = collections.Counter(s1)
        valid = 0 # 记录 window 中已经有多少字符符合要求了

        while right < len(s2):
            cur_right = s2[right]
            window[cur_right] += 1

            if window[cur_right] == target[cur_right]:
                valid += 1 # 如果 window 中 cur_right 字符的数量符合要求了，那么 valid += 1

            right += 1

            while left <= right and valid >= len(target): # 如果 window 中包含了所有字符，那么就开始收缩 left 指针
                if right - left == len(s1): # 如果当前的窗口大小等于 s1 的长度，则说明找到了一个排列
                    return True

                cur_left = s2[left]
                window[cur_left] -= 1
                if cur_left in target: # 如果 window 中 cur_left 字符的数量不再符合要求了，那么 valid -= 1
                    if window[cur_left] < target[cur_left]:
                        valid -= 1
                left += 1

        return False
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个字符串。
- 空间复杂度：O(n)，因为我们使用了一个 window 来存储当前的子串。

#### [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

这道题的描述是找到 s 中所有 p 的排列，我们可以使用 sliding window 来解决。

test cases:

```text
Input: s: "cbaebabacd" p: "abc"
Output: [0, 6]

Input: s: "abab" p: "ab"
Output: [0, 1, 2]
```

这道题的思路是，我们使用一个 window 来存储当前的子串，然后不断地增加 right 指针来扩大窗口，直到窗口中包含了 p 的所有字符，此时我们就需要收缩 left 指针来缩小窗口，直到窗口中不再包含 p 的所有字符。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        left, right = 0, 0
        window = collections.Counter()
        target = collections.Counter(p)
        valid = 0 # 记录 window 中已经有多少字符符合要求了
        res = []

        while right < len(s):
            cur_right = s[right]
            window[cur_right] += 1

            if window[cur_right] == target[cur_right]:
                valid += 1 # 如果 window 中 cur_right 字符的数量符合要求了，那么 valid += 1

            right += 1

            while left <= right and valid >= len(target): # 如果 window 中包含了所有字符，那么就开始收缩 left 指针
                if right - left == len(p): # 如果当前的窗口大小等于 p 的长度，则说明找到了一个排列
                    res.append(left)

                cur_left = s[left]
                window[cur_left] -= 1
                if cur_left in target: # 如果 window 中 cur_left 字符的数量不再符合要求了，那么 valid -= 1
                    if window[cur_left] < target[cur_left]:
                        valid -= 1
                left += 1

        return res
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个字符串。
- 空间复杂度：O(n)，因为我们使用了一个 window 来存储当前的子串。

#### [2444. Count Subarrays With Fixed Bounds] (https://leetcode.com/problems/count-subarrays-with-fixed-ends/)

这道题的描述是找到所有的子数组，这些子数组的最大值和最小值都在给定的范围内。

test cases:

```text
Input: nums = [2, 1, 4, 3], left = 2, right = 3
Output: 3

Input: nums = [2, 9, 2, 5, 6], left = 2, right = 8
Output: 7
```

这道题的思路是，我们使用一个 window 来存储当前的子串，然后不断地增加 right 指针来扩大窗口，直到窗口中包含了所有符合要求的子数组，此时我们就需要收缩 left 指针来缩小窗口，直到窗口中不再包含所有符合要求的子数组。

```python
class Solution:
    def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
        left, right = 0, 0
        window = []
        res = 0

        while right < len(nums):
            cur_right = nums[right]
            if left <= cur_right <= right:
                window.append(cur_right)

            right += 1

            while left <= right and (not window or max(window) > right or min(window) < left):
                if window:
                    res += len(window)
                window.pop(0)
                left += 1

        return res
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个数组。
- 空间复杂度：O(n)，因为我们使用了一个 window 来存储当前的子串。

#### [30. Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)

这道题的描述是找到 s 中所有 words 的排列，我们可以使用 sliding window 来解决。

test cases:

```text
Input: s: "barfoothefoobarman" words: ["foo", "bar"]
Output: [0, 9]

Input: s: "wordgoodgoodgoodbestword" words: ["word", "good", "best", "word"]
Output: []
```

这道题的思路是，我们使用一个 window 来存储当前的子串，然后不断地增加 right 指针来扩大窗口，直到窗口中包含了 words 的所有字符，此时我们就需要收缩 left 指针来缩小窗口，直到窗口中不再包含 words 的所有字符。

```python
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        left, right = 0, 0
        window = collections.Counter()
        target = collections.Counter(words)
        valid = 0 # 记录 window 中已经有多少字符符合要求了
        res = []

        while right < len(s):
            cur_right = s[right]
            window[cur_right] += 1

            if window[cur_right] == target[cur_right]:
                valid += 1 # 如果 window 中 cur_right 字符的数量符合要求了，那么 valid += 1

            right += 1

            while left <= right and valid >= len(target): # 如果 window 中包含了所有字符，那么就开始收缩 left 指针
                if right - left == len(words) * len(words[0]): # 如果当前的窗口大小等于 words 的长度，则说明找到了一个排列
                    res.append(left)

                cur_left = s[left]
                window[cur_left] -= 1
                if cur_left in target: # 如果 window 中 cur_left 字符的数量不再符合要求了，那么 valid -= 1
                    if window[cur_left] < target[cur_left]:
                        valid -= 1
                left += 1

        return res
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个字符串。
- 空间复杂度：O(n)，因为我们使用了一个 window 来存储当前的子串。
