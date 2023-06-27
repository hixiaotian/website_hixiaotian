### 前言

在这个专题，我们将主要讨论 stack 的具体用法。

有很多类型题会使用到 stack，比如说括号匹配，计算器等等。

### 什么是 stack

stack 是一种数据结构，它的特点是先进后出，后进先出。

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

这道题的描述就是括号匹配，我们可以使用 stack 来解决。

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
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储括号。

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

这道题的描述逆向波兰表达式，那么为什么可以使用 stack 呢？仔细思考一下，我们可以发现，逆向波兰表达式的计算顺序是先计算后面的，然后再计算前面的，这样才能保证计算的正确性。

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

- 时间复杂度：O(n)，原因是我们需要遍历整个 tokens。
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

#### [155. Min Stack](https://leetcode.com/problems/min-stack/)

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

这道题的描述是最小栈，我们可以使用 stack 来解决的原因是，我们需要先进后出，这样才能保证最小值的正确性。

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
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

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

这道题的描述是简化路径，我们可以使用 stack 来解决的原因是，我们需要先进后出，这样才能保证路径的正确性。

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

- 时间复杂度：O(n)，原因是我们需要遍历整个 path。
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储路径。

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

这道题的描述是解码字符串，我们可以使用 stack 来解决的原因是，我们需要先进后出，这样才能保证字符串的正确性。

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
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储字符串。

### Calculator

#### 模版

实际上，针对所有的计算器问题（包含括号，加减乘除），我们都可以使用这个模版来解决。

```python
class Solution:
    def calculate(self, s):
        def update(op, v):
            if op == "+": stack.append(v)
            if op == "-": stack.append(-v)
            if op == "*": stack.append(stack.pop() * v)           #for BC II and BC III
            if op == "/": stack.append(int(stack.pop() / v))      #for BC II and BC III

        it, num, stack, sign = 0, 0, [], "+"

        while it < len(s):
            if s[it].isdigit():
                num = num * 10 + int(s[it])
            elif s[it] in "+-*/":
                update(sign, num)
                num, sign = 0, s[it]
            elif s[it] == "(":                                        # For BC I and BC III
                num, j = self.calculate(s[it + 1:])
                it = it + j
            elif s[it] == ")":                                        # For BC I and BC III
                update(sign, num)
                return sum(stack), it + 1
            it += 1
        update(sign, num)
        return sum(stack)
```

上面能够解决所有的计算器问题的原因是，就是使用几个步骤：

1. 遇到数字，就把数字加入到数字中。
2. 遇到符号，就把数字加入到 stack 中。
3. 遇到左括号，就把数字加入到 stack 中。
4. 遇到右括号，就把数字加入到 stack 中。

不过如果你想要解决的是基本计算器，你也可以使用不 recursive 的方法来解决。

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

这道题的描述是基本计算器，我们可以使用 stack 来解决的原因是，我们需要先进后出，这样才能保证计算的正确性。

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
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

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

这道题的描述是基本计算器 II，我们可以使用 stack 来解决的原因是，我们需要先进后出，这样才能保证计算的正确性。

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
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

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

这道题的描述是基本计算器 III，我们可以使用 stack 来解决的原因是，我们需要先进后出，这样才能保证计算的正确性。

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
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

### 单调栈

#### [1475. Final Prices With a Special Discount in a Shop](https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/)

这道题的描述是给定一个数组，我们需要找到数组中比当前元素小的第一个元素，然后将当前元素减去这个元素，如果没有比当前元素小的元素，那么就不减。

test case:

```text
Input: prices = [8,4,6,2,3]
Output: [4,2,4,2,3]
Explanation:
For item 0 with price[0]=8 you will receive a discount equivalent to prices[1]=4, therefore, the final price you will pay is 8 - 4 = 4.
For item 1 with price[1]=4 you will receive a discount equivalent to prices[3]=2, therefore, the final price you will pay is 4 - 2 = 2.
For item 2 with price[2]=6 you will receive a discount equivalent to prices[3]=2, therefore, the final price you will pay is 6 - 2 = 4.
For items 3 and 4 you will not receive any discount at all.
```

很显然，想要实现 O（n）的时间复杂度，我们需要使用单调栈。注意，栈里存储的是 index，而不是 value。

我们用前面的例子来理解一下单调递增栈：

```text
prices = [8,4,6,2,3]
stack = []
index = 0, value = 8

stack = [0]
index = 1, value = 4

这时候发现栈顶元素比当前元素大，（8 > 4) 那么我们就将栈顶元素弹出，直到栈顶元素比当前元素小，或者栈为空，再将当前元素压入栈中。
把弹出的元素减去当前元素，这样完成了第一个更新。
prices[0] -= prices[1] = 4

stack = [1]
index = 2, value = 6

stack = [1, 2]
index = 3, value = 2

这时候发现栈顶元素比当前元素大，（4，6 > 2） 那么我们就将栈顶元素弹出，直到栈顶元素比当前元素小，或者栈为空，再将当前元素压入栈中。
把弹出的元素减去当前元素，这样完成了第二个更新。
prices[2] -= prices[3] = 4
prices[1] -= prices[3] = 2

...
```

这样我们就能找到比当前元素小的第一个元素，然后将当前元素减去这个元素。

```python
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        stack = []
        for index, value in enumerate(prices):
            # 如果当前元素比栈顶元素小，那么就将栈顶元素减去当前元素
            while stack and prices[stack[-1]] >= value:
                prices[stack.pop()] -= value
            stack.append(index)

        return prices
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个数组。
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

#### [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

这道题的描述是每日温度，去查找下一个比当前温度高的温度的距离。

test case:

```text
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

我们用前面的例子来理解一下单调递减栈（这里用单调递减是因为我们需要找到下一个比当前温度高的温度）：

```text
temperatures = [73,74,75,71,69,72,76,73]
stack = []

index = 0, value = 73
stack = [0]

index = 1, value = 74
当我们发现栈顶元素比当前元素小的时候 (73 < 74)，我们就将栈顶元素弹出，直到栈顶元素比当前元素大，或者栈为空，再将当前元素压入栈中。
同时，我们记录下当前元素的 index 与栈顶元素的 index 的差值，这样就能得到下一个比当前温度高的温度的距离。
更新后：
result[0] = 1 - 0 = 1
stack = [1]

index = 2, value = 75
当我们发现栈顶元素比当前元素小的时候 (74 < 75)，我们就将栈顶元素弹出，直到栈顶元素比当前元素大，或者栈为空，再将当前元素压入栈中。
同时，我们记录下当前元素的 index 与栈顶元素的 index 的差值，这样就能得到下一个比当前温度高的温度的距离。
更新后：
result[1] = 2 - 1 = 1
stack = [2]

index = 3, value = 71
当我们发现栈顶元素比当前元素小的时候 (75 > 71)，我们就将当前元素压入栈中。
stack = [2, 3]

index = 4, value = 69
当我们发现栈顶元素比当前元素小的时候 (71 > 69)，我们就将当前元素压入栈中。
stack = [2, 3, 4]

index = 5, value = 72
当我们发现栈顶元素比当前元素小的时候 (69 < 72)，我们就将栈顶元素弹出，直到栈顶元素比当前元素大，或者栈为空，再将当前元素压入栈中。
同时，我们记录下当前元素的 index 与栈顶元素的 index 的差值，这样就能得到下一个比当前温度高的温度的距离。
更新后：
result[4] = 5 - 4 = 1
result[3] = 5 - 3 = 2
stack = [2, 5]

index = 6, value = 76
当我们发现栈顶元素比当前元素小的时候 (72 < 76)，我们就将栈顶元素弹出，直到栈顶元素比当前元素大，或者栈为空，再将当前元素压入栈中。
同时，我们记录下当前元素的 index 与栈顶元素的 index 的差值，这样就能得到下一个比当前温度高的温度的距离。
更新后：
result[5] = 6 - 5 = 1
result[2] = 6 - 2 = 4
stack = [6]

index = 7, value = 73
当我们发现栈顶元素比当前元素小的时候 (76 > 73)，我们就将当前元素压入栈中。
stack = [6, 7]
```

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        answer = [0] * n
        stack = []

        for curr_day, curr_temp in enumerate(temperatures):
            # Pop until the current day's temperature is not
            # warmer than the temperature at the top of the stack
            while stack and temperatures[stack[-1]] < curr_temp:
                prev_day = stack.pop()
                answer[prev_day] = curr_day - prev_day
            stack.append(curr_day)

        return answer
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个数组。
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

#### [901. Online Stock Span](https://leetcode.com/problems/online-stock-span/)

这道题的描述是股票价格跨度，我们需要计算出每一天的股票价格跨度。

test case:

```text
Input
["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
[[], [100], [80], [60], [70], [60], [75], [85]]
Output
[null, 1, 1, 1, 2, 1, 4, 6]

Explanation
StockSpanner stockSpanner = new StockSpanner();
stockSpanner.next(100); // return 1
stockSpanner.next(80);  // return 1
stockSpanner.next(60);  // return 1
stockSpanner.next(70);  // return 2
stockSpanner.next(60);  // return 1
stockSpanner.next(75);  // return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
stockSpanner.next(85);  // return 6
```

```python
class StockSpanner:

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        res = 1
        while self.stack and self.stack[-1][0] <= price:
            res += self.stack.pop()[1]
        self.stack.append((price, res))
        return res
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个数组。
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

#### [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)

这道题的描述是下一个更大的元素 II，去查找下一个比当前元素大的元素。

test case:

```text
Input: nums = [1,2,1]
Output: [2,-1,2]
```

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stack = []
        res = [-1] * len(nums)
        for i in range(len(nums)):
            while stack and nums[i] > nums[stack[-1]]:
                res[stack[-1]] = nums[i]
                stack.pop()
            stack.append(i)
        for i in range(len(nums)):
            while stack and nums[i] > nums[stack[-1]]:
                res[stack[-1]] = nums[i]
                stack.pop()
        return res
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个数组。
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

#### [32. Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)

这道题的描述是最长有效括号，去查找最长有效括号。

test case:

```text
Input: s = "(()"
Output: 2

Input: s = ")()())"
Output: 4
```

我们用栈来解决这个问题，我们首先将 -1 放入栈中。然后，对于遇到的每个 '(' ，我们将它的下标放入栈中。对于遇到的每个 ')' ，我们弹出栈顶的元素并将当前元素的下标与弹出元素下标作差，得出当前有效括号字符串的长度。通过这种方法，我们继续计算有效子字符串的长度，并最终返回最长有效子字符串的长度。

我们看一个例子：

```text
s = ")()())"

stack = [-1]

index = 0, value = )
第一个字符是')'，我们将其下标放入栈中。
因为栈为空，我们将')'下标放入栈中。
stack = [0]

index = 1, value = (
第二个字符是'('，我们将其下标放入栈中。
stack = [0, 1]

index = 2, value = )
第三个字符是')'，我们弹出栈顶元素并将当前元素的下标与弹出元素下标作差，得出当前有效括号字符串的长度。
stack = [0]
res = max(res, i - stack[-1]) = max(0, 2 - 0) = 2

index = 3, value = (
第四个字符是'('，我们将其下标放入栈中。
stack = [0, 3]

index = 4, value = )
第五个字符是')'，我们弹出栈顶元素并将当前元素的下标与弹出元素下标作差，得出当前有效括号字符串的长度。
stack = [0]
res = max(res, i - stack[-1]) = max(1, 4 - 0) = 4

index = 5, value = )
第六个字符是')'，我们弹出栈顶元素并将当前元素的下标与弹出元素下标作差，得出当前有效括号字符串的长度。
stack = []
因为栈为空，我们将当前元素的下标放入栈中。
stack = [5]

```

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1] # 这里是-1，因为如果第一个字符是')'，那么就会出现stack为空的情况，导致后面的计算出错
        res = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    res = max(res, i - stack[-1])
        return res
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个数组。
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

#### [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

这道题的描述是柱状图中最大的矩形，去查找柱状图中最大的矩形。

test case:

```text
Input: heights = [2,1,5,6,2,3]
Output: 10
```

我们用前面的例子来理解一下单调递增栈（这里用单调递增是因为我们需要找到下一个比当前柱子矮的柱子）：

```text

index = 0, value = 2
stack = [0]

index = 1, value = 1
当我们发现栈顶元素比当前元素大的时候 (2 > 1)，我们就将栈顶元素弹出，直到栈顶元素比当前元素小，或者栈为空，再将当前元素压入栈中。
同时我们要更新最大面积，最大面积 = 当前柱子高度 * (当前柱子的 index - 栈顶元素的 index)，这里的 index 是指柱子在数组中的下标。
更新后：
result = 2 * 1 = 2 （栈为空，说明前面的柱子都比当前柱子矮，所以最大面积就是当前柱子的高度 * 当前柱子的 index）
stack = [1]

index = 2, value = 5
当我们发现栈顶元素比当前元素小的时候 (1 < 5)，我们就将当前元素压入栈中。
stack = [1, 2]

index = 3, value = 6
当我们发现栈顶元素比当前元素小的时候 (5 < 6)，我们就将当前元素压入栈中。
stack = [1, 2, 3]

index = 4, value = 2
当我们发现栈顶元素比当前元素大的时候 (6 > 2)，我们就将栈顶元素弹出，直到栈顶元素比当前元素小，或者栈为空，再将当前元素压入栈中。
同时我们要更新最大面积，最大面积 = 当前柱子高度 * (当前柱子的 index - 栈顶元素的 index - 1)，这里的 index 是指柱子在数组中的下标。
更新后：
result = 6 * (4 - 2 - 1）= 6 （此时栈里是 [1, 2]，所以最大面积就是当前柱子的高度 * (4 - 2 - 1)）
result = 5 * (4 - 1 - 1）= 10 （此时栈里是 [1]，所以最大面积就是当前柱子的高度 * (4 - 1 - 1)）
stack = [1, 4]

index = 5, value = 3
当我们发现栈顶元素比当前元素小的时候 (2 < 3)，我们就将当前元素压入栈中。
stack = [1, 4, 5]

index = 6, value = 0
当我们发现栈顶元素比当前元素大的时候 (3 > 0)，我们就将栈顶元素弹出，直到栈顶元素比当前元素小，或者栈为空，再将当前元素压入栈中。
同时我们要更新最大面积，最大面积 = 当前柱子高度 * (当前柱子的 index - 栈顶元素的 index - 1)，这里的 index 是指柱子在数组中的下标。
更新后：
result = 3 * (6 - 4 - 1）= 3 （此时栈里是 [1, 4]，所以最大面积就是当前柱子的高度 * (6 - 4 - 1)）
result = 2 * (6 - 1 - 1）= 8 （此时栈里是 [1]，所以最大面积就是当前柱子的高度 * (6 - 1 - 1)）
result = 1 * (6 - 0 - 1）= 6 （此时栈里是 []，所以最大面积就是当前柱子的高度 * (6)）
stack = [6]
```

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        res = 0
        heights.append(0) # 这里是为了让最后一个柱子也能被处理
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                res = max(res, h * w)
            stack.append(i)
        return res
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个数组。
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

#### [85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)

这道题的描述是最大矩形，去查找最大矩形。

test case:

```text
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 6
```

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        res = 0
        heights = [0] * len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                heights[j] = heights[j] + 1 if matrix[i][j] == "1" else 0
            res = max(res, self.largestRectangleArea(heights))
        return res

    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        res = 0
        heights.append(0)
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                res = max(res, h * w)
            stack.append(i)
        return res
```

复杂度分析：

- 时间复杂度：O(nm)，原因是我们需要遍历整个数组。
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。

#### [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

这道题的描述是接雨水，去计算接雨水的量。

test case:

```text
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []
        res = 0
        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                bottom = height[stack.pop()]
                if not stack:
                    break
                h = min(height[i], height[stack[-1]]) - bottom
                w = i - stack[-1] - 1
                res += h * w
            stack.append(i)
        return res
```

复杂度分析：

- 时间复杂度：O(n)，原因是我们需要遍历整个数组。
- 空间复杂度：O(n)，原因是我们需要使用 stack 来存储数字。
