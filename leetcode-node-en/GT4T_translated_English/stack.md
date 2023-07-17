### Foreword

In this topic, we will focus on the specific use of stack.

There are many types of questions that use stack, such as bracket matching, calculator and so on.

### What is stack?

A stack is a data structure characterized by first in, last out, and last in, first out.

Why does it apply to parentheses matching? Because when parentheses are matched, we need to match the parentheses first and then match the parentheses, so as to ensure that the matching of parentheses is correct.

Why does it apply to calculators? Because the calculation order of the calculator is to multiply and divide first and then add and subtract, so as to ensure the correctness of the calculation.

### Basic questions

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

The description of this problem is bracket matching, and we can use stack to solve it.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire string.
- Space complexity: O (n). The reason is that we need to use stack to store the brackets.

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

The description of this problem is a reverse Polish expression, so why can stack be used? If we think about it carefully, we can find that the calculation order of the reverse Polish expression is to calculate the latter first and then the former, so as to ensure the correctness of the calculation.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire token.
- Space complexity: O (n), because we need to use stacks to store numbers.

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

The description of this problem is the minimum stack. The reason why we can use stack to solve it is that we need first-in last-out to ensure the correctness of the minimum value.

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

Complexity analysis:

- Time complexity: O (1), because we only need to operate on the top of the stack.
- Space complexity: O (n), because we need to use stacks to store numbers.

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

The description of this problem is to simplify the path. The reason why we can use stack to solve it is that we need first-in, last-out, so as to ensure the correctness of the path.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire path.
- Space complexity: O (n), because we need to use stack to store paths.

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

The description of this problem is to decode the string. The reason why we can use stack to solve it is that we need to first in, last out, so as to ensure the correctness of the string.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire string.
- Space complexity: O (n), because we need to use stacks to store strings.

### Calculator

#### Template

In fact, we can use this template to solve all calculator problems (including parentheses, addition, subtraction, multiplication and division).

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

The reason the above can solve all the calculator problems is to use a few steps:

1. When you meet a number, you add the number to the number.
2. When a symbol is encountered, the number is added to the stack.
3. When an open parenthesis is encountered, the number is added to the stack.
4. When a closing parenthesis is encountered, the number is added to the stack.

However, if you want to solve the basic calculator, you can also use the non-recursive method to solve it.

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

The description of this problem is a basic calculator, and the reason why we can use stack to solve it is that we need first in, last out, so as to ensure the correctness of the calculation.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire string.
- Space complexity: O (n), because we need to use stacks to store numbers.

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

The description of this problem is Basic Calculator II. The reason why we can use stack to solve it is that we need first in, last out, so that we can ensure the correctness of the calculation.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire string.
- Space complexity: O (n), because we need to use stacks to store numbers.

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

The description of this problem is Basic Calculator III. The reason why we can use stack to solve it is that we need first in, last out, so that we can ensure the correctness of the calculation.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire string.
- Space complexity: O (n), because we need to use stacks to store numbers.

### Monotone stack

#### [1475. Final Prices With a Special Discount in a Shop](https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/)

The description of this problem is that given an array, we need to find the first element in the array that is smaller than the current element, and then subtract this element from the current element. If there is no element smaller than the current element, then it will not be subtracted.

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

Obviously, to achieve O (n) time complexity, we need to use a monotone stack. Note that the stack stores the index, not the value.

Let's use the previous example to understand the monotonically increasing stack:

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

This allows us to find the first element that is smaller than the current element, and then subtract this element from the current element.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire array.
- Space complexity: O (n), because we need to use stacks to store numbers.

#### [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

The description of this question is the daily temperature, to find the distance of the next temperature higher than the current temperature.

test case:

```text
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

Let's use the previous example to understand the monotonically decreasing stack (here we use monotonically decreasing because we need to find the next temperature higher than the current temperature):

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire array.
- Space complexity: O (n), because we need to use stacks to store numbers.

#### [901. Online Stock Span](https://leetcode.com/problems/online-stock-span/)

The description of this problem is the stock price span, and we need to calculate the stock price span for each day.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire array.
- Space complexity: O (n), because we need to use stacks to store numbers.

#### [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)

The description of this problem is the next larger element II, to find the next element larger than the current element.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire array.
- Space complexity: O (n), because we need to use stacks to store numbers.

#### [32. Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)

The description of this question is the longest valid parenthesis. Go to find the longest valid parenthesis.

test case:

```text
Input: s = "(()"
Output: 2

Input: s = ")()())"
Output: 4
```

We use the stack to solve this problem, and we first put -1 on the stack. Then, for each ' (' encountered, we put its subscript on the stack. For each ')' encountered, we pop the element at the top of the stack and subtract the current element's subscript from the pop-up element's subscript to find the length of the current valid parenthesis string. In this way, we continue to calculate the length of the valid substring and eventually return the length of the longest valid substring.

Let's look at an example:

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire array.
- Space complexity: O (n), because we need to use stacks to store numbers.

#### [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

The description of this question is the largest rectangle in the histogram. Find the largest rectangle in the histogram.

test case:

```text
Input: heights = [2,1,5,6,2,3]
Output: 10
```

Let's use the previous example to understand the monotonically increasing stack (we use monotonically increasing here because we need to find the next column that is shorter than the current column):

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire array.
- Space complexity: O (n), because we need to use stacks to store numbers.

#### [85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)

The description of this problem is the largest rectangle. Go to find the largest rectangle.

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

Complexity analysis:

- Time complexity: O (nm), because we need to traverse the entire array.
- Space complexity: O (n), because we need to use stacks to store numbers.

#### [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

The description of this question is to receive rainwater and calculate the amount of rainwater.

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

Complexity analysis:

- Time complexity: O (n), because we need to traverse the entire array.
- Space complexity: O (n), because we need to use stacks to store numbers.
