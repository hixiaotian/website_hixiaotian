### Foreword

We use this topic to discuss the backtracking algorithm, which is a recursive algorithm that traverses a tree depth-first (because it traverses the whole tree, the time complexity is generally not low).

The difference between the backtracking algorithm and the DFS algorithm is that the backtracking algorithm will judge each node in the process of traversal. If it does not meet the conditions, it will backtrack to the previous node and then traverse it. Let's take the simplest example to distinguish when to use the backtracking algorithm and when to use the DFS algorithm.

For example, we have an array, each element in the array is a choice, and we need to choose an element that meets the condition from these choices. Then we can use the backtracking algorithm, because in the process of traversing, we can judge each element, if it does not meet the conditions, we will backtrack to the previous element, and then traverse.

But if we want to select all the eligible elements from these choices, then we need to use the DFS algorithm, because we need to traverse the whole tree and find all the eligible elements.

The idea of the backtracking algorithm is very simple and can be roughly divided into the following steps:

1. Path: make a choice
2. Select: Make a selection from the selection list
3. Condition: If the end condition is met, the selection is added to the result.
4. Undo: Undo the selection

The pseudocode is as follows:

```python
result = []
def backtrack(path, selectionList):
    if meet the end condition:
        result.add(path)
        return

    for selection in selectionList:
        make a choice
        backtrack(path, selectionList)
        undo the choice
```

### Basic questions

#### [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

The description of this problem is to give a string containing only the numbers 2-9 and return all the letter combinations it can represent. Answers can be returned in any order. The mapping of numbers to letters is given as follows (same as phone keys). Note that 1 does not correspond to any letter.

test cases:

```text
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Input: digits = ""
Output: []
```

The idea of this problem is the idea of backtracking algorithm. We can put the letters corresponding to each number into an array, and then backtrack the array, and finally get all the results. Specifically, every phone number can be seen as a tree, for example, 2 can be seen as such a tree, and the corresponding branch is the letter corresponding to 2. Then we can backtrack the tree and get all the results.

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        # use a dictionary to store the letters corresponding to each number
        dic = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }

        res = []

        def backtrack(index, path):
            if index == len(digits): # if the index is equal to the length of the input, it means that the result has been traversed
                res.append("".join(path))
                return

            possible = dic[digits[index]]

            for letter in possible:
                path.append(letter) # make a choice
                backtrack(index + 1, path) # enter the next level
                path.pop() # undo the choice

        backtrack(0, [])
        return res
```

Complexity analysis:

- Time complexity: O (3 ^ _ 4 ^ n), where m is the number of digits corresponding to 3 letters in the input (including digits 2, 3, 4, 5, 6 and 8), n is the number of digits corresponding to 4 letters in the input (including digits 7 and 9), and m + n is the total number of input digits. When the input contains m numbers corresponding to 3 letters and n numbers corresponding to 4 letters, the total number of different letter combinations is 3 ^ m _ m 4 ^ n), each letter combination needs to be traversed.
- Space complexity: O (m + n), where m is the number of digits corresponding to 3 letters in the input, n is the number of digits corresponding to 4 letters in the input, and m + n is the total number of input digits. In addition to the return value, the space complexity mainly depends on the hash table and the number of recursive calls in the backtracking process. The size of the hash table is independent of the input and can be regarded as a constant. The maximum number of recursive calls is m + n.

#### [39. Combination Sum](https://leetcode.com/problems/combination-sum/)

Given an array of candidates with no repeating elements and a target number target, the description of this problem is to find all the combinations in candidates that can make the sum of numbers target. Numbers in candidates can be selected repeatedly without restriction.

test cases:

```text
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]

Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
```

The idea of this problem is also the idea of backtracking algorithm. We can regard every number as a tree. For example, when candidates = [2, 3, 6, 7] and target = 7, we can regard it as such a tree:

```text
                2
        /       |       \
       2        3        6
      /  \     /  \     /  \
    2     3   2    3   2    3
```

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def backtrack(index, path, rest):
            if rest == 0: # if the target is 0, it means that the result has been traversed
                res.append(path[:])
                return

            if rest < 0: # if the target is less than 0, it means that the result is not feasible
                return

            for i in range(index, len(candidates)):
                path.append(candidates[i]) # make a choice
                backtrack(i, path, rest - candidates[i]) # enter the next level
                path.pop() # undo the choice

        backtrack(0, [], target)
        return res
```

Complexity analysis:

- Time complexity: O (S), where S is the sum of the lengths of all feasible solutions. In the worst case, all combinations are feasible, for example, candidates = [1, 1, 1, 1,1,1,1,1, 1,1,1,1], target = 11. In this case, there are O (2 ^ N) combinations, and each combination is O (N) long, so the time complexity is O (N \ \* 2 ^ N). In the best case, the number in the combination is 1, and the time complexity is O (N).
- Space complexity: O (target). In addition to the answer array, the space complexity depends on the stack depth of the recursion, which, in the worst case, requires O (target) layers of recursion.

#### [40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)

Given an array candidates and a target number target, the description of this problem is to find all the combinations in candidates that can make the sum of numbers target. Each number in candidates can be used only once in each combination.

test cases:

```text
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: [[1,1,6],[1,2,5],[1,7],[2,6]]

Input: candidates = [2,5,2,1,2], target = 5
Output: [[1,2,2],[5]]
```

The difference between this question and the previous one is that the numbers in the array in the previous question can be selected repeatedly without restriction, while the numbers in the array in this question can only be used once in each combination. Then we can make some changes to this problem, for example, when recursing, we can add one to the index, so that we can ensure that each number will only be used once.

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def backtrack(start, path, rest):
            if rest == 0: # if the target is 0, it means that the result has been traversed
                res.append(path[:])
                return

            if rest < 0: # if the target is less than 0, it means that the result is not feasible
                return

            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    # if the current number is equal to the previous number, it means that it has been traversed, so return directly
                    # For example, candidates = [1,1,2,5,6,7,10], target = 8
                    # When i = 1, path = [1], rest = 7, at this time i > start and candidates[i] == candidates[i - 1]
                    # So skip directly without recursion
                    continue

                path.append(candidates[i]) # make a choice
                backtrack(i + 1, path, rest - candidates[i]) # enter the next level
                path.pop() # undo the choice

        candidates.sort() # sort the array
        backtrack(0, [], target)
        return res
```

Complexity analysis:

- Time complexity: O (S), where S is the sum of the lengths of all feasible solutions. In the worst case, all combinations are feasible, for example, candidates = [1, 1, 1, 1,1,1,1,1, 1,1,1,1], target = 11. In this case, there are O (2 ^ N) combinations, and each combination is O (N) long, so the time complexity is O (N \ \* 2 ^ N). In the best case, the number in the combination is 1, and the time complexity is O (N).
- Space complexity: O (target). In addition to the answer array, the space complexity depends on the stack depth of the recursion, which, in the worst case, requires O (target) layers of recursion.

#### [216. Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)

The description of this problem is to give an integer K and an integer n, and find all the combinations of the number of K whose sum is n. Only positive integers from 1 to 9 are allowed in the combination, and there are no duplicate numbers in each combination.

test cases:

```text
Input: k = 3, n = 7
Output: [[1,2,4]]

Input: k = 3, n = 9
Output: [[1,2,6],[1,3,5],[2,3,4]]

Input: k = 4, n = 1
Output: []

Input: k = 3, n = 2
Output: []

Input: k = 9, n = 45
Output: [[1,2,3,4,5,6,7,8,9]]
```

The difference between this question and the previous one is that the numbers in the array in the previous question can be selected repeatedly without restriction, while the numbers in the array in this question can only be used once in each combination. Then we can make some changes to this problem, for example, when recursing, we can add one to the index, so that we can ensure that each number will only be used once.

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []

        def backtrack(start, path, rest):
            if rest == 0 and len(path) == k: # if the target is 0 and the length of the path is k, it means that the result has been traversed
                res.append(path[:])
                return

            if rest < 0 or len(path) >= k: # if the target is less than 0 or the length of the path is greater than or equal to k, it means that the result is not feasible
                return

            for i in range(start, 10):
                path.append(i) # make a choice
                backtrack(i + 1, path, rest - i) # enter the next level
                path.pop() # undo the choice

        backtrack(1, [], n)
        return res
```

Complexity analysis:

- Time complexity: O (9!), where 9! = 362880。
- Space complexity: O (K), where K is the length of the combination.

#### [46. Permutations](https://leetcode.com/problems/permutations/)

The description of this problem is that given an array nums containing no repeated numbers, all possible permutations of these numbers are returned. You can return the answers in any order.

test cases:

```text
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Input: nums = [0,1]
Output: [[0,1],[1,0]]

Input: nums = [1]
Output: [[1]]
```

The idea of this problem is also the idea of backtracking algorithm. We can regard every number as a tree. For example, when nums = [1,2,3], we can regard it as such a tree:

```text
                1
        /       |       \
       2        3        2
      /  \     /  \     /  \
    3     2   2    1   1    3
```

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []

        def backtrack(path):
            if len(path) == len(nums): # if the length of the path is equal to the length of the array, it means that the result has been traversed
                res.append(path[:])
                return

            for num in nums:
                if num in path: # if the number is already in the path, it means that it has been traversed, so skip directly
                    continue

                path.append(num) # make a choice
                backtrack(path) # enter the next level
                path.pop() # undo the choice

        backtrack([])
        return res
```

In fact, this problem can also be solved using python packages, such as the permutations function in itertools, which can return all combinations of an array, such as:

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        return list(itertools.permutations(nums))
```

Complexity analysis:

- Time complexity: O (N \ \* N!), where N is the length of the array.
- Space complexity: O (N \ \* N!), where N is the length of the array.

#### [47. Permutations II](https://leetcode.com/problems/permutations-ii/)

test cases:

```text
Input: nums = [1,1,2]
Output: [[1,1,2],[1,2,1],[2,1,1]]

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

The difference between this question and the previous one is that the numbers in the array in this question can be repeated. Then we can use counter to record the number of times each number appears, and then add or subtract the corresponding number before and after the backtracking.

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        results = []
        def backtrack(path, counter):
            if len(path) == len(nums):
                # if the length of the path is equal to the length of the array, it means that the result has been traversed
                results.append(path[:])
                return

            for num in counter:
                if counter[num] > 0: # if the number of times the number appears is greater than 0, it means that it has not been traversed
                    path.append(num)
                    counter[num] -= 1 # counter minus one
                    backtrack(path, counter)
                    path.pop() # undo the choice
                    counter[num] += 1 # counter plus one

        backtrack([], Counter(nums))
        return results
```

Or we can use the visited array to record whether each number has been visited or not, so that we don't need to use counter.

```python
def permuteUnique(self, nums: List[int]) -> List[List[int]]:

    def backtrack(res, cur, nums, visited):
        if len(cur) == len(nums):
            res.append(cur[:])

        for i in range(len(nums)):
            if (visited[i] == 1) or (i > 0 and nums[i] == nums[i - 1] and visited[i - 1] == 0):
                continue

            visited[i] = 1
            cur.append(nums[i])
            backtrack(res, cur, nums, visited)
            visited[i] = 0
            cur.pop()
        return res

    res = []
    boolean = [0 for _ in range(len(nums))]
    nums.sort()
    return backtrack(res, [], nums, boolean)
```

In fact, this problem can also be solved using python packages, such as the permutations function in itertools, which can return all combinations of an array, such as:

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        return list(set(itertools.permutations(nums)))
```

Complexity analysis:

- Time complexity: O (N \ \* N!), where N is the length of the array.
- Space complexity: O (N \ \* N!), where N is the length of the array.

#### [78. Subsets](https://leetcode.com/problems/subsets/)

test cases:

```text
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

Input: nums = [0]
Output: [[],[0]]
```

The description of this problem is to give an integer array nums without repeated elements and return all possible subsets (power sets) of the array. The solution set cannot contain duplicate subsets.

The idea of this problem is also the idea of backtracking algorithm. We can regard every number as a tree. For example, when nums = [1,2,3], we can see that the subset tree is such a tree:

```text
                []
        /       |       \
       1        2        3
      /  \     /  \     /  \
    2     3   3    1   1    2
   / \   / \ / \  / \ / \  / \
  3   1 1   2  2 1  3 3  2 2  1
```

In fact, this is wrong, because there are repeated subsets in this tree, such as [1,2] and [2,1], so we need to prune the tree. The way to prune is to pass a start parameter when we recurse. This parameter represents the subscript of the current number. For example, when we traverse to 2, The start is 1, so we only need to iterate through the numbers after 2, so there are no duplicate subsets.

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []

        def backtrack(start, path):
            res.append(path[:])

            for i in range(start, len(nums)):
                path.append(nums[i]) # make a choice
                backtrack(i + 1, path) # start from the next number
                path.pop() # undo the choice

        backtrack(0, [])
        return res
```

In fact, this problem can also be solved using python packages, such as the combinations function in itertools, which can return all combinations of an array, such as:

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        return chain.from_iterable(chain([combinations(nums, i) for i in range(len(nums) + 1)]))
```

Complexity analysis:

- Time complexity: O (N \ \* 2 ^ N), where N is the length of the array.
- Space complexity: O (N \ \* 2 ^ N), where N is the length of the array.

#### [90. Subsets II](https://leetcode.com/problems/subsets-ii/)

test cases:

```text
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

Input: nums = [0]
Output: [[],[0]]
```

The difference between this question and the previous one is that the array of this question contains duplicate elements, so we need to prune the tree. The pruning method is to pass in a start parameter when recursing. This parameter represents the subscript of the current number. For example, when we traverse to 2, start is 1. Then we only need to go through the numbers after 2, so that there will be no duplicate subsets. And it should be noted that when we traverse, if the current number is equal to the previous number, it means that we have traversed and returned directly.

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []

        def backtrack(start, path):
            res.append(path[:])

            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]: # if the current number is equal to the previous number, it means that we have traversed and returned directly
                    continue

                path.append(nums[i]) # make a choice
                backtrack(i + 1, path) # start from the next number
                path.pop() # undo the choice

        nums.sort() # sort the array
        backtrack(0, [])
        return res
```

In fact, this problem can also be solved using python packages, such as the combinations function in itertools, which can return all combinations of an array, such as:

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        return set([tuple(sorted(item)) for item in chain.from_iterable(chain([combinations(nums, i) for i in range(len(nums) + 1)]))])
```

Complexity analysis:

- Time complexity: O (N \ \* 2 ^ N), where N is the length of the array.
- Space complexity: O (N \ \* 2 ^ N), where N is the length of the array.

### Advanced questions

#### [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)

test cases:

```text
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Input: n = 1
Output: ["()"]
```

The description of this problem is that given an integer n, all valid combinations consisting of n pairs of brackets are generated. A valid combination needs to satisfy: the left parenthesis must be closed in the correct order.

The idea of this problem is also the idea of backtracking algorithm. We can regard each bracket as a tree. For example, when n = 3, we can regard it as such a tree:

![image](https://leetcode.com/problems/generate-parentheses/Figures/22/5.png)

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []

        def backtrack(left, right, path):
            if left == 0 and right == 0: # if the number of left and right parentheses is 0, it means that the result is obtained
                res.append("".join(path))
                return

            if left > right: # if the number of left parentheses is greater than the number of right parentheses, it means that the result is not obtained
                return

            if left > 0: # if the number of left parentheses is greater than 0, you can add a left parenthesis
                path.append("(")
                backtrack(left - 1, right, path)
                path.pop()

            if right > 0: # if the number of right parentheses is greater than 0, you can add a right parenthesis
                path.append(")")
                backtrack(left, right - 1, path)
                path.pop()

        backtrack(n, n, [])
        return res
```

Complexity analysis:

- Time complexity: O (4 ^ n/sqrt (n)). During backtracking, each answer takes O (n) time to copy into the answer array.
- Space complexity: O (n). In addition to the answer array, the space we need depends on the depth of the recursion stack. Each layer of recursion function needs O (1) space. At most 2 n layers of recursion, so the space complexity is O (n).

#### [79. Word Search](https://leetcode.com/problems/word-search/)

test cases:

```text
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false
```

The description of this problem is given a two-dimensional grid and a word, find out whether the word exists in the grid. Words must be formed alphabetically by letters in adjacent cells, where "adjacent" cells are those that are horizontally or vertically adjacent. Letters in the same cell are not allowed to be reused.

The idea of this problem is also the idea of backtracking algorithm. We can regard each letter as a tree, such as board = [ [ "A", "B", "C", "E"], [ "S", 'F', 'C', 'S'], [ "A",'D ','E', E ']]. When word = "ABCCED", we can see such a tree:

```text
                A
        /       |       \
       B        S        D
      /  \     /  \     /  \
    C     D   F    E   E    C
   / \   / \ / \  / \ / \  / \
  C   E C   C C   S E   E C   C
```

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def backtrack(i, j, index):
            if index == len(word): # if the index is equal to the length of the word, it means that the result is obtained
                return True

            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[index]:
                # if the index is out of bounds or the current letter is not equal to the letter in the word, it means that the result is not obtained
                return False

            board[i][j] = "#" # make the current letter invalid
            res = backtrack(i + 1, j, index + 1) or backtrack(i - 1, j, index + 1) or backtrack(i, j + 1, index + 1) or backtrack(i, j - 1, index + 1) # backtracking
            board[i][j] = word[index] # restore the current letter
            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if backtrack(i, j, 0):
                    return True

        return False
```

Complexity analysis:

- Time complexity: O (M \ _ N \ _ 4 ^ L), where M and N are the height and width of the two-dimensional grid, respectively, and L is the length of the string word. Every time the function backtrack is called, except for the first time we can enter four branches, the rest of the time we will enter up to three branches (because each position can only be used once, so the branches that come through can not go back). Since the word length is L, the time complexity of the backtracking function is O (3 ^ L). And we want to perform O (M \ _ N) times, so the total time complexity is O (M \ _ N \ \* 3 ^ L).
- Space complexity: O (L), where L is the length of the string word. Stack space for mostly recursive calls.

#### [93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)

test cases:

```text
Input: s = "25525511135"
Output: ["255.255.11.135","255.255.111.35"]

Input: s = "0000"
Output: ["0.0.0.0"]

Input: s = "101023"
Output: ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
```

The description of this problem is to give a string s containing only numbers and return all the valid IP addresses that can be obtained from s. You can return the answers in any order.

The idea of this problem is also the idea of backtracking algorithm, we can regard every number as a tree, for example, when s = "25525511135", we can regard it as such a tree:

```text
                2
        /       |       \
       5        5        5
      /  \     /  \     /  \
    5     5   5    5   5    5
   / \   / \ / \  / \ / \  / \
  2   5 2   5 2   5 2   5 2   5
```

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []

        def backtrack(start, path):
            if start == len(s) and len(path) == 4:
                # if the start index is equal to the length of the string s and the length of the path is equal to 4, it means that the result is obtained
                res.append(".".join(path))
                return

            if len(path) > 4: # if the length of the path is greater than 4, it means that the result is not obtained
                return

            for i in range(start, len(s)):
                if s[start] == "0" and i > start:
                    # if the first letter of the current number is 0 and the length of the current number is greater than 1, it means that the result is not obtained
                    return

                num =  int(s[start:i + 1])

                if num > 255: # if the current number is greater than 255, it means that the result is not obtained
                    return

                path.append(str(num)) # make the current number valid
                backtrack(i + 1, path) # backtracking
                path.pop() # restore the current number

        backtrack(0, [])
        return res
```

Complexity analysis:

- Time complexity: O (1), because the length of the IP address is fixed.
- Space complexity: O (1), because the length of the IP address is fixed.

#### [131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/)

test cases:

```text
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]

Input: s = "a"
Output: [["a"]]
```

The description of this problem is that given a string s, divide s into some substrings, so that each substring is a palindrome string. Returns s all possible partitioning schemes.

The idea of this problem is also the idea of backtracking algorithm. We can regard every letter as a tree. For example, when s = "AAB", we can regard it as such a tree:

```text
                a
        /       |       \
       a        a        b
      /  \     /  \     /  \
    a     b   a    b   a    b
   / \   / \ / \  / \ / \  / \
  b   a b   a b   a b   a b   a
```

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []

        def backtrack(index, path):
            if index == len(s): # if the index is equal to the length of the string s, it means that the result is obtained
                res.append(path[:])
                return

            for i in range(index, len(s)):
                if self.isPalindrome(s[index:i + 1]): # if the current string is a palindrome string
                    path.append(s[index:i + 1]) # make the current string valid
                    backtrack(i + 1, path) # backtracking
                    path.pop() # restore the current string

        backtrack(0, [])
        return res

    def isPalindrome(self, s: str) -> bool:
        return s == s[::-1]
```

Complexity analysis:

- Time complexity: O (N \ \* 2 ^ N), where N is the length of the string.
- Space complexity: O (N \ \* 2 ^ N), where N is the length of the string.

#### [254. Factor Combinations](https://leetcode.com/problems/factor-combinations/)

test cases:

```text
Input: n = 1
Output: []

Input: n = 37
Output: []

Input: n = 12
Output: [[2,6],[2,2,3],[3,4]]

Input: n = 32
Output: [[2,16],[2,2,8],[2,2,2,4],[2,2,2,2,2],[2,4,4],[4,8]]
```

The description of this problem is to give an integer n and return all possible combinations of the factors of n. A factor is a number that is divisible by another number.

The idea of this problem is also the idea of backtracking algorithm. We can regard every number as a tree. For example, when n = 12, we can regard it as such a tree:

```text
                12
        /       |       \
       2        3        4
      /  \     /  \     /  \
    2     6   3    4   4    3
   / \   / \ / \  / \ / \  / \
  3   4 2   3 2   2 3   2 2   2
```

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        res = []

        def backtrack(index, path, target):
            if target == 1 and len(path) > 1: # if the target is equal to 1 and the length of the path is greater than 1, it means that the result is obtained
                res.append(path[:])
                return

            for i in range(index, target + 1):
                if target % i == 0: # if the current number is a factor of the target
                    path.append(i) # make the current number valid
                    backtrack(i, path, target // i) # backtracking
                    path.pop() # restore the current number

        backtrack(2, [], n)
        return res
```

Complexity analysis:

- Time complexity: O (N \ \* log N), where N is the size of the integer n.
- Space complexity: O (N \ \* log N), where N is the size of the integer n.

#### [401. Binary Watch](https://leetcode.com/problems/binary-watch/)

test cases:

```text
Input: turnedOn = 1
Output: ["0:01","0:02","0:04","0:08","0:16","0:32","1:00","2:00","4:00","8:00"]

Input: turnedOn = 9
Output: []
```

The description of this problem is given a non-negative integer turnedOn, which represents the number of LEDs currently on, and returns all possible times. You can return the answers in any order.

The idea of this problem is also the idea of backtracking algorithm. We can regard every number as a tree. For example, when turnedOn = 1, we can regard it as such a tree:

```text
                0
        /       |       \
       1        2        4
      /  \     /  \     /  \
    2     4   3    5   5    6
   / \   / \ / \  / \ / \  / \
  3   5 4   6 5   7 6   8 7   9
```

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        res = []

        def backtrack(index, path, hour, minute):
            if hour > 11 or minute > 59:
                return

            if turnedOn == 0:
                res.append(str(hour) + ":" + "0" * (minute < 10) + str(minute))
                return

            for i in range(index, 10):
                if i < 4:
                    backtrack(i + 1, path, hour + (1 << i), minute)
                else:
                    backtrack(i + 1, path, hour, minute + (1 << (i - 4)))

        backtrack(0, [], 0, 0)
        return res
```

Complexity analysis:

- Time complexity: O (1).
- Space complexity: O (1).

#### [526. Beautiful Arrangement](https://leetcode.com/problems/beautiful-arrangement/)

test cases:

```text
Input: n = 2
Output: 2

Input: n = 1
Output: 1
```

The description of this problem is to give a positive integer n and return all possible beautiful permutations of n numbers. An array is considered to be a beautiful permutation if its ith element is divisible by I.

The idea of this problem is also the idea of backtracking algorithm. We can regard every number as a tree. For example, when n = 2, we can regard it as such a tree:

```text
                1
        /       |       \
       2        1        2
```

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def countArrangement(self, n: int) -> int:
        res = []

        def backtrack(index, path):
            if index == n + 1:
                res.append(path[:])
                return

            for i in range(1, n + 1):
                if i in path:
                    continue

                if i % index == 0 or index % i == 0:
                    path.append(i)
                    backtrack(index + 1, path)
                    path.pop()

        backtrack(1, [])
        return len(res)
```

Complexity analysis:

- Time complexity: O (K), where K is the number of eligible permutations.
- Space complexity: O (n), where n is the size of a positive integer n.

#### [784. Letter Case Permutation](https://leetcode.com/problems/letter-case-permutation/)

test cases:

```text
Input: s = "a1b2"
Output: ["a1b2","a1B2","A1b2","A1B2"]

Input: s = "3z4"
Output: ["3z4","3Z4"]
```

The description of this problem is to give a string s and return all possible upper and lower case letters. Letters are case-sensitive.

The idea of this problem is also the idea of backtracking algorithm. We can regard every letter as a tree. For example, when s = "a1b2", we can regard it as such a tree:

```text
                a
        /       |       \
       a        A        a
      /  \     /  \     /  \
    a     b   A    b   a    b
   / \   / \ / \  / \ / \  / \
  b   2 B   2 b   2 B   2 b   2
```

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        res = []

        def backtrack(index, path):
            if index == len(s): # if the index is equal to the length of the string, it means that we have traversed all the characters in the string
                res.append("".join(path))
                return

            if s[index].isdigit(): # if the current character is a number, we can only continue to traverse
                path.append(s[index]) # make a choice
                backtrack(index + 1, path) # enter the next level of decision tree
                path.pop() # cancel the choice
            else: # if the current character is a letter, we can choose to convert it to uppercase or lowercase
                path.append(s[index].lower()) # make a choice
                backtrack(index + 1, path) # enter the next level of decision tree
                path.pop() # cancel the choice
                path.append(s[index].upper()) # make a choice
                backtrack(index + 1, path) # enter the next level of decision tree
                path.pop() # cancel the choice

        backtrack(0, [])
        return res
```

Complexity analysis:

- Time complexity: O (N \ \* 2 ^ N), where N is the length of the string.
- Space complexity: O (N \ \* 2 ^ N), where N is the length of the string.

#### [797. All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/)

test cases:

```text
Input: graph = [[1,2],[3],[3],[]]
Output: [[0,1,3],[0,2,3]]

Input: graph = [[4,3,1],[3,2,4],[3],[4],[]]
Output: [[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]

Input: graph = [[1],[]]
Output: [[0,1]]

Input: graph = [[1,2,3],[2],[3],[]]
Output: [[0,1,2,3],[0,2,3],[0,3]]

Input: graph = [[1,3],[2],[3],[]]
Output: [[0,1,2,3],[0,3]]
```

The description of this problem is given a directed acyclic graph with n nodes, find all the paths from 0 to n-1 and output them (not in order).

The idea of this problem is also the idea of backtracking algorithm. We can regard each node as a tree. For example, when graph = [ [1, 2], [3], [3], []], we can regard it as such a tree:

```text
                0
        /       |       \
       1        3        3
      /  \     /  \     /  \
    2     3   3    3   3    3
   / \   / \ / \  / \ / \  / \
  3   3 3   3 3   3 3   3 3   3
```

Then we can backtrack the tree and get all the results.

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        res = []

        def backtrack(index, path):
            if index == len(graph) - 1: # if the index is equal to the length of the graph minus 1, it means that we have traversed all the nodes in the graph
                res.append(path[:])
                return

            for i in graph[index]:
                path.append(i) # make a choice
                backtrack(i, path) # enter the next level of decision tree
                path.pop() # cancel the choice

        backtrack(0, [0])
        return res
```

Complexity analysis:

- Time complexity: O (N \ \* 2 ^ N), where N is the length of the graph.
- Space complexity: O (N \ \* 2 ^ N), where N is the length of the graph.

#### [842. Split Array into Fibonacci Sequence](https://leetcode.com/problems/split-array-into-fibonacci-sequence/)

test cases:

```text
Input: "123456579"
Output: [123,456,579]

Input: "11235813"
Output: [1,1,2,3,5,8,13]

Input: "112358130"
Output: []

Input: "0123"
Output: []

Input: "1101111"
Output: [110, 1, 111]
```

The description of this problem is that given a numeric string S, such as S = "123456579", we can divide it into multiple Fibonacci sequences, such as [123, 456, 579]. A Fibonacci sequence is a sequence in which each number is the sum of the two preceding numbers. Formally, given a Fibonacci sequence, we want to delete at least one digit from it so that the remaining digits form a strictly increasing sequence. Returns all possible cases.

The idea of this problem is also the idea of backtracking algorithm, we can regard every number as a tree, for example, when S = "123456579", we can regard it as such a tree:

```text
                1
        /       |       \
       2        2        2
      /  \     /  \     /  \
    3     3   3    3   3    3
   / \   / \ / \  / \ / \  / \
  4   4 4   4 4   4 4   4 4   4
```

Then we can backtrack the tree and get all the results.

```python
未完待续
```

#### [126. Word Ladder II](https://leetcode.com/problems/word-ladder-ii/)

test cases:

```text
Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output:
[
  ["hit","hot","dot","dog","cog"],
  ["hit","hot","lot","log","cog"]
]

Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: []
```

The description of this problem is to find all the shortest conversion sequences from begin word to end word given two words (beginWord and endWord) and a dictionary wordList. The conversion follows the following rules:

- Only one letter can be changed per conversion.
- The intermediate word in the conversion must be a dictionary word.

Of course we can use bfs to do it, everything is stored in the queue, but this will MLE:

```python
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList):

        if not endWord or not beginWord or not wordList or endWord not in wordList or beginWord == endWord:
            return []

        graph = collections.defaultdict(list)
        for word in wordList:
            for i in range(len(beginWord)):
                graph[word[:i] + "*" + word[i+1:]].append(word)
        print(graph)

        ans = []
        q = collections.deque([(beginWord, [beginWord])])
        visited = set()
        visited.add(beginWord)

        while q and not ans:
            length = len(q)
            localvisited = set()
            for _ in range(length):
                word, path = q.popleft()
                for i in range(len(beginWord)):
                    candidate = word[:i] + "*" + word[i+1:]
                    for nxt in graph[candidate]:
                        if nxt == endWord:
                            ans.append(path + [endWord])
                        if nxt not in visited:
                            localvisited.add(nxt)
                            q.append((nxt, path + [nxt]))

            visited = visited.union(localvisited)
        return ans

```

The better idea of this problem is also the idea of backtracking algorithm. We can think of every word as a tree, such as beginWord = "hit", endWord = "cog". When wordList = [ "hot", "dot", "dog", "lot", 'log', "cog"], we can see a tree like this:

```text
                hit
        /       |       \
       hot      dot      lot
      /  \     /  \     /  \
    dot   lot dot  lot dot  lot
   / \   / \ / \  / \ / \  / \
  dog cog dog cog dog cog dog cog
```

Then we can build the graph with BFS first, and then backtrack the graph to get all the results.

```python
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList):
        wordList.append(beginWord) # needs to add begin word into list for indexing (L126 already consider endword to be in the wordList)
        indexes = self.build_indexes(wordList)
        distance = self.bfs(endWord, indexes)
        results = []
        self.dfs(beginWord, endWord, distance, indexes, [beginWord], results)
        return results
    def build_indexes(self, wordList):
        indexes = {}
        for word in wordList:
            for i in range(len(word)):
                key = word[:i] + '%' + word[i + 1:]
                if key in indexes:
                    indexes[key].add(word)
                else:
                    indexes[key] = set([word])
        return indexes
    def bfs(self, end, indexes): # bfs from end to start
        distance = {end: 0}
        queue = deque([end])
        while queue:
            word = queue.popleft()
            for next_word in self.get_next_words(word, indexes):
                if next_word not in distance:
                    distance[next_word] = distance[word] + 1
                    queue.append(next_word)
        return distance
    def get_next_words(self, word, indexes):
        words = set()
        for i in range(len(word)):
            key = word[:i] + '%' + word[i + 1:]
            for w in indexes.get(key, []):
                words.add(w)
        return words

    def dfs(self, curt, target, distance, indexes, path, results):
        if curt == target:
            results.append(list(path))
            return
        for word in self.get_next_words(curt, indexes):
            if word not in distance: # if there is no a possible way in word ladder
                return
            if distance[word] != distance[curt] - 1:
                continue
            path.append(word)
            self.dfs(word, target, distance, indexes, path, results)
            path.pop()
```
