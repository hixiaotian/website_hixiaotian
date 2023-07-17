### Foreword

In this topic, we mainly discuss the specific usage of sliding window.

You can use sliding windows for many types of problems, such as finding the longest non-repeating substring in a string, or finding the longest contiguous subarray in an array.

Let's briefly introduce the template of sliding window:

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

The core idea of this template is:

- 1. First, increase the right pointer continuously to enlarge the window.
- 2. When the window contains all the characters of t, it starts to shrink the left pointer until it gets the smallest window.

### Basic questions

#### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

The description of this problem is to find the longest non-repeating substring, and we can use sliding window to solve it.

test cases:

```text
Input: s = "abcabcbb"
Output: 3

Input: s = "bbbbb"
Output: 1

Input: s = "pwwkew"
Output: 3
```

The idea of this problem is that we use a window to store the current substring, and then increase the right pointer to expand the window until there are duplicate characters in the window. At this time, we need to shrink the left pointer to shrink the window until there are no more duplicate characters in the window.

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

            while window[cur_right] > 1: # 如果 window 中 cur_right 字符的数量大于 1，说明窗口中出现了重复的字符
                cur_left = s[left]
                window[cur_left] -= 1
                left += 1

            res = max(res, right - left)

        return res
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire string.
- Space complexity: O (n), because we use a window to store the current substring.

#### [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

The description of this problem is to find the smallest substring containing t, and we can use sliding window to solve it.

test cases:

```text
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"

Input: s = "a", t = "a"
Output: "a"
```

The idea of this problem is that we use a window to store the current substring, and then increase the right pointer to expand the window until the window contains all the characters of t. At this time, we need to shrink the left pointer to shrink the window until the window no longer contains all the characters of t.

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
                if cur_left in target: # 如果 window 中 cur_left 字符的数量不再符合要求了，那么 valid -= 1
                    if window[cur_left] == target[cur_left]:
                        valid -= 1
                window[cur_left] -= 1
                left += 1

        return s[min_start: min_start + min_length] if min_length != float("inf") else ""
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire string.
- Space complexity: O (n), because we use a window to store the current substring.

#### [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)

The description of this problem is to find the smallest contiguous subarray such that the sum of the subarrays is greater than or equal to s. We can use sliding window to solve this problem.

test cases:

```text
Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2

Input: s = 4, nums = [1,4,4]
Output: 1

Input: s = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0
```

The idea of this problem is that we use a window to store the current substring, and then increase the right pointer to expand the window until the sum in the window is greater than or equal to s. At this time, we need to shrink the left pointer to shrink the window until the sum in the window is less than s.

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        left, right = 0, 0
        res = float("inf")
        window = 0 # 记录 window 中的和

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

- Time complexity: O (n), because we need to traverse the entire array.
- Space complexity: O (1), because we use a window to store the current substring.

#### [567. Permutation in String](https://leetcode.com/problems/permutation-in-string/)

The description of this problem is to determine whether S2 contains the permutation of S1, and we can use sliding window to solve it.

test cases:

```text
Input: s1 = "ab", s2 = "eidbaooo"
Output: True

Input: s1= "ab", s2 = "eidboaoo"
Output: False
```

The idea of this problem is that we use a window to store the current substring, and then increase the right pointer to expand the window until the window contains all the characters of S1. At this time, we need to shrink the left pointer to shrink the window until the window no longer contains all the characters of S1.

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
            right += 1

            if cur_right in target:
                if window[cur_right] == target[cur_right]:
                    valid += 1 # 如果 window 中 cur_right 字符的数量符合要求了，那么 valid += 1

            while right - left >= len(s1): # 如果 window 中包含了所有字符，那么就开始收缩 left 指针
                if valid == len(target): # 如果当前的窗口大小等于 s1 的长度，则说明找到了一个排列
                    return True

                cur_left = s2[left]
                if cur_left in target: # 如果 window 中 cur_left 字符的数量不再符合要求了，那么 valid -= 1
                    if window[cur_left] == target[cur_left]:
                        valid -= 1
                window[cur_left] -= 1
                left += 1
        return False
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire string.
- Space complexity: O (n), because we use a window to store the current substring.

#### [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

The description of this problem is to find all the permutations of p in s, and we can use sliding window to solve it.

test cases:

```text
Input: s: "cbaebabacd" p: "abc"
Output: [0, 6]

Input: s: "abab" p: "ab"
Output: [0, 1, 2]
```

The idea of this problem is that we use a window to store the current substring, and then increase the right pointer to expand the window until the window contains all the characters of p. At this time, we need to shrink the left pointer to shrink the window until the window no longer contains all the characters of p.

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

            if cur_right in target:
                if window[cur_right] == target[cur_right]:
                    valid += 1

            right += 1

            while right - left >= len(p): # 如果 window 中包含了所有字符，那么就开始收缩 left 指针
                if valid == len(target): # 如果当前的窗口大小等于 p 的长度，则说明找到了一个排列
                    res.append(left)

                cur_left = s[left]

                if cur_left in target: # 如果 window 中 cur_left 字符的数量不再符合要求了，那么 valid -= 1
                    if window[cur_left] == target[cur_left]:
                        valid -= 1

                window[cur_left] -= 1

                left += 1

        return res
```

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire string.
- Space complexity: O (n), because we use a window to store the current substring.

### Advanced questions

#### [30. Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)

The description of this problem is to find the arrangement of all words in s, and we can use sliding window to solve it.

test cases:

```text
Input: s: "barfoothefoobarman" words: ["foo", "bar"]
Output: [0, 9]

Input: s: "wordgoodgoodgoodbestword" words: ["word", "good", "best", "word"]
Output: []
```

The idea of this problem is that we use a window to store the current substring, and then increase the right pointer to expand the window until the window contains all the characters of words. At this time, we need to shrink the left pointer to shrink the window until the window no longer contains all the characters of words.

```python
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        word_len = len(words[0])
        all_word_len = word_len * len(words)
        target = collections.Counter(words)
        res = []

        for i in range(word_len):
            word_dict = collections.defaultdict(int)
            left = right = i

            while right < len(s):
                word = s[right:right + word_len]
                word_dict[word] += 1

                if right - left + word_len == all_word_len:
                    if word_dict == target:
                        res.append(left)

                    word = s[left:left + word_len]
                    word_dict[word] -= 1
                    if word_dict[word] == 0:
                        del word_dict[word]

                    left += word_len
                right += word_len
        return res
```

Complexity analysis:

- Time complexity: O (nm). The reason is that we need to traverse the entire string, where n is the length of s and m is the length of words.
- Space complexity: O (n), because we use a window to store the current substring.

#### [2444. Count Subarrays With Fixed Bounds] (https://leetcode.com/problems/count-subarrays-with-fixed-bounds/)

To be continued
