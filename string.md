Let's start with some basic (not that basic) operations on String by Python:
0. Stack!!!

You should be aware that <b>Most of the String problems</b> is associate with stack!

That's right! sometimes we tend to use operations quite similar to compiler, thus we need to be familiar with everything in <b>STACK</b>

Stack operations
```python
stack = []
# push into stack
stack.append(1)  # [1]
stack.append(2)  # [1, 2]
stack.append(3)  # [1, 2, 3]
# pop out the top element
top = stack.pop() # top = 3, stack -> [1, 2]
# get the top element
top = stack[-1] # top = 2, stack -> [1, 2]  (similar to peek)
```

But in fact, how we use python queue in leetcode problem is that:  
a) We use head-tail linked list (deque):
```python
from collections import deque
de1 = deque()
de1.append(1) # [1]
de1.append(2) # [1, 2]
de1.appendleft(5) # [5, 1, 2]
de1.appendleft(6) # [6, 5, 1, 2]
print(de1.count(1)) # 1 appears 1
de1.reverse() # [2, 1, 5, 6]
de1.extend([444, 555, 666]) # [2, 1, 5, 6, 444, 555, 666]
de1.pop() # [2, 1, 5, 6, 444, 555]
de1.popleft() # [1, 5, 6, 444, 555]
```

b) Priority queue:
```python
import queue
q = queue.Queue()
q.put(3)
q.put(5)
q.put(4)
full = q.full() # True
while not q.empty():
    print(q.get()) # [3, 5, 4]

# min first
pq = queue.PriorityQueue()
pq.put(3)
pq.put(5)
pq.put(4)
while not pq.empty():
    print(pq.get()) # [3, 4, 5]

# max first
pq = []
pq.append(3)
pq.append(5)
pq.append(4)
pq.sort(reverse=1)
while pq:
    print(pq.pop())
```

By the way, we should know how to implement a queue through <b>2 stack</b>, which will be push = O(1) and pop = O(1)

This is the problem of Leetcode 232:
```python
class MyQueue:
    def __init__(self):
        self.LI = []  # Last In
        self.FO = []  # First Out

    def push(self, x):
        self.LI.append(x)
    
    def pop(self):
        if not self.FO:
            while self.LI:
                self.FO.append(self.LI.pop())
        return self.FO.pop()
        
    def peek(self):
        if not self.FO:
            return self.LI[0]
        else:
            return self.FO[-1]

    def empty(self):
        return not self.LI and not self.FO
```

Or if we want one stack to represent queue, it is also OK!
```python 
class MyQueue(object):
    def __init__(self):
        self.st = []

    def push(self, x):
        if len(self.st) == 0:
            self.st.append(x)
            return
        tmp = self.st.pop(-1)
        self.push(x)
        self.st.append(tmp)

    def pop(self):
        return self.st.pop(-1)

    def peek(self):
        return self.st[-1]

    def empty(self):
        return len(self.st) == 0
```



1. Reverse String:

```python
ori_str = "abcd"
reverse_str = ori_str[::-1] #come to "dcba"
```

2. Find SubString:
```python
super_str = "abcde"
sub_str = "bc"
not_sub_str = "abd"
pos = super_str.find(sub_str) #pos = 1, find from beginning
pos_2 = super_str.rfind(sub_str) # pos_2 = 1, find from the rear
pos_3 = super_str.find(not_sub_str) # pos_2 = -1, not substring found
```
3. index & count
```python
ori_str = "abcde"
pos = ori_str.index("c") # pos = 2
pos_1 = ori_str.rindex("b") # pos_1 = 1
```
4. count
```python
ori_str = "abcdeaa"
num = ori_str.count("a") # num = 3
num_1 = ori_str.count("a", 0, 3) # num_1 = 1
```

5. Split, Partition, Split with line
```python
str = "123123123"               
str_list = str.split('2') # ['1', '31', '31', '3']          
str_tuple = str.partition('2')  # ('1', '2', '3123123')     

str='abc\nabc\nabc\nabc'  
str_list = str.splitlines() # ['abc', 'abc', 'abc', 'abc']
```
6. Judgement
```python
str0='0123abcd'
str1='12345'
str2='abcdef'
str3='    '

str0.startswith('0')    # check if start with 0, True
str0.endswith('0')      # check if end with 0, False

str1.isalnum()        # check if the str all consist of alphabet and number, True
str2.isalpha()        # check if the str all consist of alphabet, True
str0.isdigit()        # check if the str all consist of digit, False
str3.isspace()    # check if all consist of space, True
```

7. ATTENTION: String cannot be changed, but we can change by..
```python
str0 = "what"
# str0[2] = "i" # This is not allowed!!!
list_str0 = list(str0)
list_str0[2] = "i"
str0 = "".join(list_str0)
print(str0)
```

8. Mapping
```python
date = "2019-8-15"
Y, M, D = map(int, date.split('-'))
# Y = 2019, M = 8, D = 15
```

---

### Leetcode

##### 242. Valid Anagram

Description: return true if ``s`` is an anagram of ``t``
```
Input: s = "anagram", t = "nagaram"
Output: true
```

Solution 1 with Collections (inner Hash Map):
```python
def isAnagram(self, s: str, t: str) -> bool:
    from collections import Counter
    # Counter(s) = Counter({'a': 3, 'n': 1, 'g': 1, 'r': 1, 'm': 1})
    return Counter(s) == Counter(t)
```

Solution 2 with HashMap (By ourself) in O(n):
```python
def isAnagram(self, s: str, t: str) -> bool:
    map1 = {}

    if s == '' and t == '':
        return True
    if len(s) != len(t):
        return False
    for itr in range(len(s)):
        map1[s[itr]] = map1.setdefault(s[itr], 0) + 1
        map1[s[itr]] = map1.setdefault(t[itr], 0) + 1
    for i in map1.values():
        if i != 0:
            return False
    return True

```

##### 409. Valid Anagram

Solution 1 with Hash Map:
```python
def longestPalindrome(self, s: str) -> int:
    map1 = {}
    for item in range(len(s)):
        map1[s[item]] = map1.setdefault(s[item], 0) + 1
    res = 0    
    for item in map1.values():
        res += (item // 2) * 2
    if res < len(s):
        res += 1
    return res
```
##### 205. Isomorphic Strings

Description:
Given "egg", "add", return true.
Given "foo", "bar", return false.
Given "paper", "title", return true.

Solution 1 with find function:
```python
def isIsomorphic(self, s, t):
    # e.g. s = "eggggwe"
    # [s.find(i) for i in s] = [0, 1, 1, 1, 1, 5, 0]
    return [s.find(i) for i in s] == [t.find(j) for j in t]
```

Solution 2 with dict function:
```python
def isIsomorphic(self, s, t):
    if len(s) != len(t): return False
    map1 = {}
    for i in range(len(s)):
        if s[i] in map1:
            if map1[s[i]] != t[i]:
                return False
        elif t[i] in map1.values():
            return False
        else:
            map1[s[i]] = t[i]
    return True
```

##### 647. Palindromic Substrings

```
Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
```


Solution 1 with find function:
```python
res = 0
def countSubstrings(self, s: str) -> int:
    for i in range(len(s)):
        self.extension(s, i, i);
        self.extension(s, i, i+1);
    return self.res

def extension(self, s: str, start: int, end: int):
    while start >= 0 and end <= len(s) - 1 and s[start] == s[end]:
        start -= 1
        end += 1
        self.res += 1
```


##### 9. Palindrome Number
```
Input: x = 121
Output: true

Input: x = -121
Output: false

Could you solve it without converting the integer to a string?
```

Solution 1 simple implement:
```python
def isPalindrome(self, x: int) -> bool:
    if x < 0 or (x > 0 and x % 10 == 0): return False
    ori = x
    res = 0
    
    while x != 0:
        digit = x % 10
        x = x // 10
        res = digit + res * 10
        
    return ori == res
```

Solution 2 with half loop:
```python
def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x > 0 and x % 10 == 0): return False
        right = 0
        while x > right:
            right = right * 10 + x % 10
            x = x // 10 
        return x == right or x == right // 10
```

##### 696. Count Binary Substrings

```
Input: "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
```

Solution 1 with group by character:
```python
    def countBinarySubstrings(self, s: str) -> int:
        groups = [1]
        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                groups.append(1)
            else:
                groups[-1] += 1

        ans = 0
        for i in range(1, len(groups)):
            ans += min(groups[i-1], groups[i])
        return ans
```

Solution 1 with linear scanning, not using a list, but use two var to store pre and cur:
```python
def countBinarySubstrings(self, s: str) -> int:
    preLen, curLen, count = 0, 1, 0
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            curLen += 1
        else:
            preLen = curLen
            curLen = 1

        if preLen >= curLen:
            count += 1
    return count
```
![rGdjF0f](http://hixiaotian.com/content/images/2021/09/rGdjF0f.png)