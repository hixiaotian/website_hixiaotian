### 前言

在这个专题，我们将主要讨论stack的具体用法。

有很多类型题会使用到stack，比如说括号匹配，计算器等等。

### 什么是stack

stack是一种数据结构，它的特点是先进后出，后进先出。

为什么适用于括号匹配呢？因为括号匹配的时候，我们需要先匹配到的括号后匹配到，这样才能保证括号的匹配是正确的。

为什么适用于计算器呢？因为计算器的计算顺序是先乘除后加减，这样才能保证计算的正确性。

### 基础题

#### [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

test cases:

```text
Input: s = "()"
Output: true

Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false

```

这道题的描述就是括号匹配，我们可以使用stack来解决。

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        dic = {")": "(", "}": "{", "]": "["}


        for i in range(len(s)):
            if s[i] in dic.keys():
                if not stack or stack[-1] != dic[s[i]]:
                    return False
                else:
                    stack.pop()
            else:
                stack.append(s[i])

        return stack == []
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个字符串。
- 空间复杂度：O(n)，原因是我们需要使用stack来存储括号。

#### [150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

test cases:

```text
Input: tokens = ["2","1","+","3","*"]
Output: 9

Input: tokens = ["4","13","5","/","+"]
Output: 6

Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
Output: 22
Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

这道题的描述逆向波兰表达式，那么为什么可以使用stack呢？仔细思考一下，我们可以发现，逆向波兰表达式的计算顺序是先计算后面的，然后再计算前面的，这样才能保证计算的正确性。

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for i in range(len(tokens)):
            if tokens[i] in ["+", "-", "*", "/"]:
                num1 = stack.pop()
                num2 = stack.pop()
                if tokens[i] == "+":
                    stack.append(num1 + num2)
                elif tokens[i] == "-":
                    stack.append(num2 - num1)
                elif tokens[i] == "*":
                    stack.append(num1 * num2)
                else:
                    stack.append(int(num2 / num1))
            else:
                stack.append(int(tokens[i]))
        return stack[-1]
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个tokens。
- 空间复杂度：O(n)，原因是我们需要使用stack来存储数字。

### [155. Min Stack](https://leetcode.com/problems/min-stack/)

test cases:

```text
Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
```

这道题的描述是最小栈，我们可以使用stack来解决的原因是，我们需要先进后出，这样才能保证最小值的正确性。

```python
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
            return
        
        current_min = self.stack[-1][1]
        self.stack.append((val, min(val, current_min)))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]        

    def getMin(self) -> int:
        return self.stack[-1][1]
```

复杂度分析：

- 时间复杂度：O(1)，原因是我们只需要操作栈顶。
- 空间复杂度：O(n)，原因是我们需要使用stack来存储数字。

#### [71. Simplify Path](https://leetcode.com/problems/simplify-path/)

test cases:

```text
Input: path = "/home/"
Output: "/home"

Input: path = "/../"
Output: "/"

Input: path = "/home//foo/"
Output: "/home/foo"

Input: path = "/a/./b/../../c/"
Output: "/c"
```

这道题的描述是简化路径，我们可以使用stack来解决的原因是，我们需要先进后出，这样才能保证路径的正确性。

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        path = path.split("/")
        for i in range(len(path)):
            if path[i] == "..":
                if stack:
                    stack.pop()
            elif path[i] != "." and path[i] != "":
                stack.append(path[i])
        return "/" + "/".join(stack)
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个path。
- 空间复杂度：O(n)，原因是我们需要使用stack来存储路径。

#### [394. Decode String](https://leetcode.com/problems/decode-string/)

test cases:

```text
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

Input: s = "3[a2[c]]"
Output: "accaccacc"

Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"

Input: s = "abc3[cd]xyz"
Output: "abccdcdcdxyz"
```

这道题的描述是解码字符串，我们可以使用stack来解决的原因是，我们需要先进后出，这样才能保证字符串的正确性。

```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        for i in range(len(s)):
            if s[i] == "]":
                string = ""
                while stack[-1] != "[":
                    string = stack.pop() + string
                stack.pop()
                num = ""
                while stack and stack[-1].isdigit():
                    num = stack.pop() + num
                stack.append(int(num) * string)
            else:
                stack.append(s[i])
        return "".join(stack)
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个字符串。
- 空间复杂度：O(n)，原因是我们需要使用stack来存储字符串。

#### [224. Basic Calculator](https://leetcode.com/problems/basic-calculator/)

test cases:

```text
Input: s = "1 + 1"
Output: 2

Input: s = " 2-1 + 2 "
Output: 3

Input: s = "(1+(4+5+2)-3)+(6+8)"
Output: 23
```

这道题的描述是基本计算器，我们可以使用stack来解决的原因是，我们需要先进后出，这样才能保证计算的正确性。

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        sign = 1
        res = 0
        i = 0
        while i < len(s):
            if s[i].isdigit():
                num = ""
                while i < len(s) and s[i].isdigit():
                    num += s[i]
                    i += 1
                res += sign * int(num)
                i -= 1
            elif s[i] == "+":
                sign = 1
            elif s[i] == "-":
                sign = -1
            elif s[i] == "(":
                stack.append(res)
                stack.append(sign)
                res = 0
                sign = 1
            elif s[i] == ")":
                res *= stack.pop()
                res += stack.pop()
            i += 1
        return res
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个字符串。
- 空间复杂度：O(n)，原因是我们需要使用stack来存储数字。

#### [227. Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/)

test cases:

```text
Input: s = "3+2*2"
Output: 7

Input: s = " 3/2 "
Output: 1

Input: s = " 3+5 / 2 "
Output: 5
```

这道题的描述是基本计算器II，我们可以使用stack来解决的原因是，我们需要先进后出，这样才能保证计算的正确性。

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        sign = "+"
        num = 0
        for i in range(len(s)):
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            if (not s[i].isdigit() and s[i] != " ") or i == len(s) - 1:
                if sign == "+":
                    stack.append(num)
                elif sign == "-":
                    stack.append(-num)
                elif sign == "*":
                    stack.append(stack.pop() * num)
                else:
                    stack.append(int(stack.pop() / num))
                sign = s[i]
                num = 0
        return sum(stack)
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个字符串。
- 空间复杂度：O(n)，原因是我们需要使用stack来存储数字。

#### [772. Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/)

test cases:

```text
Input: s = "1 + 1"
Output: 2

Input: s = " 6-4 / 2 "
Output: 4

Input: s = "2*(5+5*2)/3+(6/2+8)"
Output: 21

Input: s = "(2+6* 3+5- (3*14/7+2)*5)+3"
Output: -12
```

这道题的描述是基本计算器III，我们可以使用stack来解决的原因是，我们需要先进后出，这样才能保证计算的正确性。

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        sign = "+"
        num = 0
        for i in range(len(s)):
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            if s[i] == "(":
                j = i + 1
                count = 1
                while count != 0:
                    if s[j] == "(":
                        count += 1
                    elif s[j] == ")":
                        count -= 1
                    j += 1
                num = self.calculate(s[i + 1:j - 1])
                i = j - 1
            if (not s[i].isdigit() and s[i] != " ") or i == len(s) - 1:
                if sign == "+":
                    stack.append(num)
                elif sign == "-":
                    stack.append(-num)
                elif sign == "*":
                    stack.append(stack.pop() * num)
                else:
                    stack.append(int(stack.pop() / num))
                sign = s[i]
                num = 0
        return sum(stack)
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个字符串。
- 空间复杂度：O(n)，原因是我们需要使用stack来存储数字。

