### Preface

In this topic, we will focus on sliding window related problems.

A lot of problem can be used to solve sliding window problems. For example, find a substring that contains all the characters in a string, find the longest substring that contains at most k distinct characters, etc.

let's introduce the template first:

```python
def slidingWindow(s: str):
    window = defaultdict(int)
    left, right = 0, 0
    while right < len(s):
        cur_right = s[right]
        window[cur_right] += 1
        right += 1
        # update window data
        ...

        # debug output
        print(f"window: [{left}, {right})")

        # shrink window
        while left < right and window needs shrink:
            cur_left = s[left]
            window[cur_left] -= 1
            if window[cur_left] == 0:
                del window[cur_left]
            left += 1
            # update window data
            ...
```

The core ideas of this template are:

1. Use two pointers to represent a window.
2. Use a hash table to record the data in the window.
3. Use the while loop to shrink the window.

#### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

This problem is to find the longest substring that contains at most k distinct characters.

test cases:

```text
Input: s = "abcabcbb"
Output: 3

Input: s = "bbbbb"
Output: 1

Input: s = "pwwkew"
Output: 3
```

The idea of this problem is to use a hash table to record the number of characters in the window, the right pointer will move forward, and the left pointer will move forward when the window needs to shrink until the window is valid. (no duplicate characters)

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

Complexity analysis:

- Time complexity: O(n), because we need to traverse the string once.
- Space complexity: O(n), because we need to store the data in the window.

#### [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

This problem is to find the minimum window that contains all the characters in the target string.

```text
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"

Input: s = "a", t = "a"
Output: "a"
```

The idea of this problem is to use a hash table to record the number of characters in the window, the right pointer will move forward, and the left pointer will move forward when the window needs to shrink until the window is valid. (contains all the characters in the target string)

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        left, right = 0, 0
        min_start, min_length = None, float("inf")
        window = collections.Counter()
        target = collections.Counter(t)
        valid = 0

        while right < len(s):
            cur_right = s[right]
            window[cur_right] += 1

            if window[cur_right] == target[cur_right]:
                valid += 1

            right += 1

            while left <= right and valid >= len(target):
                if right - left < min_length:
                    min_start = left
                    min_length = right - left

                cur_left = s[left]
                window[cur_left] -= 1
                if cur_left in target:
                    if window[cur_left] < target[cur_left]:
                        valid -= 1
                left += 1

        return s[min_start: min_start + min_length] if min_length != float("inf") else ""
```

Complexity analysis:

- Time complexity: O(n), because we need to traverse the string once.
- Space complexity: O(n), because we need to store the data in the window.

#### [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)

This problem is to find the minimum window that the sum of the elements in the window is greater than or equal to the target.

test cases:

```text
Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2

Input: s = 4, nums = [1,4,4]
Output: 1

Input: s = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0
```

The idea of this problem is to use a hash table to record the number of characters in the window, the right pointer will move forward, and the left pointer will move forward when the window needs to shrink until the window is valid. (the sum of the elements in the window is greater than or equal to the target)

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

Complexity analysis:

- Time complexity: O(n), because we need to traverse the string once.
- Space complexity: O(n), because we need to store the data in the window.

#### [567. Permutation in String](https://leetcode.com/problems/permutation-in-string/)

This problem is to find whether the string s2 contains the permutation of s1.

test cases:

```text
Input: s1 = "ab", s2 = "eidbaooo"
Output: True

Input: s1= "ab", s2 = "eidboaoo"
Output: False
```

The idea of this problem is to use a hash table to record the number of characters in the window, the right pointer will move forward, and the left pointer will move forward when the window needs to shrink until the window is valid. Now we need to check whether the window contains the permutation of s1.

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        left, right = 0, 0
        window = collections.Counter()
        target = collections.Counter(s1)
        valid = 0

        while right < len(s2):
            cur_right = s2[right]
            window[cur_right] += 1

            if window[cur_right] == target[cur_right]:
                valid += 1

            right += 1

            while left <= right and valid >= len(target):
                if right - left == len(s1):
                    return True

                cur_left = s2[left]
                window[cur_left] -= 1
                if cur_left in target:
                    if window[cur_left] < target[cur_left]:
                        valid -= 1
                left += 1

        return False
```

Complexity analysis:

- Time complexity: O(n), because we need to traverse the string once.
- Space complexity: O(n), because we need to store the data in the window.

#### [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

This problem is to find all the anagrams of s in the string p.

test cases:

```text
Input: s: "cbaebabacd" p: "abc"
Output: [0, 6]

Input: s: "abab" p: "ab"
Output: [0, 1, 2]
```

The idea of this problem is to use a hash table to record the number of characters in the window, the right pointer will move forward, and the left pointer will move forward when the window needs to shrink until the window is valid. Now we need to check whether the window contains the anagrams of p.

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        left, right = 0, 0
        window = collections.Counter()
        target = collections.Counter(p)
        valid = 0
        res = []

        while right < len(s):
            cur_right = s[right]
            window[cur_right] += 1

            if window[cur_right] == target[cur_right]:
                valid += 1

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
