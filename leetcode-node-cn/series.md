## 类型题

这里包含了很多典型系列题，比如买卖股票系列 I, II, III, IV 等等。在这里我统一称之为类型题，因为它们都是一类题目，解法也是类似的。

### 买卖股票

#### [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

这道题的描述是：给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。注意你不能在买入股票前卖出股票。

Test cases:

```text
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
```

这道题的解法是：遍历数组，记录当前最小值，然后计算当前值与最小值的差值，与当前最大差值比较，如果大于当前最大差值，则更新最大差值。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price, max_profit = prices[0], float('-inf')

        for i in range(len(prices)):
            min_price = min(prices[i], min_price)
            max_profit = max(max_profit, prices[i] - min_price)

        return max_profit
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次数组
- 空间复杂度：O(1)，使用常数个变量

#### [122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

这道题的描述是：给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

Test cases:

```text
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
```

这道题的解法是：遍历数组，如果当前值大于前一个值，则计算当前值与前一个值的差值，加到最大利润上。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0

        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                max_profit += prices[i] - prices[i - 1]

        return max_profit
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次数组
- 空间复杂度：O(1)，使用常数个变量

#### [123. Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)

这道题的描述是：给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成两笔交易。

Test cases:

```text
Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
```

这道题的解法是：动态规划。定义四个变量，分别表示第一次买入、第一次卖出、第二次买入、第二次卖出的最大利润。遍历数组，更新这四个变量。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        def get_profit(start, end):
            if start > end:
                return 0

            temp_min = prices[start]
            max_profit = 0
            for i in range(start, end + 1):
                temp_min = min(temp_min, prices[i])
                max_profit = max(max_profit, prices[i] - temp_min)

            return max_profit



        if not prices or len(prices) == 1:
            return 0

        max_profit = 0
        for i in range(len(prices) + 1):
            left_max = get_profit(0, i - 1)
            right_max = get_profit(i, len(prices) - 1)
            max_profit = max(max_profit, left_max + right_max)

        return max_profit
```

这个方法是 O(N^2)的时间复杂度，会超时。下面是一个解释更清楚的方法：

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        firstCost = prices[0]
        firstTransanctionProfit = 0
        mostMoneyInhand = -prices[0]
        mostProfit = 0
        for cur in prices:
            firstCost = min(firstCost, cur)
            firstTransanctionProfit = max(firstTransanctionProfit, cur-firstCost)

            mostMoneyInhand = max(mostMoneyInhand, firstTransanctionProfit-cur)
            mostProfit = max(mostProfit, mostMoneyInhand + cur)
        return mostProfit
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次数组
- 空间复杂度：O(1)，使用常数个变量

#### [188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)

这道题的描述是：给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。

Test cases:

```text
Input: k = 2, prices = [2,4,1]
Output: 2

Input: k = 2, prices = [3,2,6,5,0,3]
Output: 7
```

这道题的解法是：动态规划。定义一个二维数组，第 i 行第 j 列表示第 i 次交易，第 j 天的最大利润。遍历数组，更新这个二维数组。

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if k == 0: return 0

        cost = [float('inf') for _ in range(k + 1)]
        profit = [0 for _ in range(k + 1)]

        for i in range(len(prices)):
            for j in range(1, k + 1):
                cost[j] = min(cost[j], prices[i] - profit[j - 1])
                profit[j] = max(profit[j], prices[i] - cost[j])

        return profit[-1]
```

复杂度分析：

- 时间复杂度：O(nk)，遍历一次数组
- 空间复杂度：O(k)，使用 k 个变量

### Jump Game

#### [55. Jump Game](https://leetcode.com/problems/jump-game/)

这道题的描述是：给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个位置。

Test cases:

```text
Input: nums = [2,3,1,1,4]
Output: true

Input: nums = [3,2,1,0,4]
Output: false
```

这道题的解法是：贪心算法。遍历数组，记录当前能到达的最远位置，如果当前位置大于最远位置，则返回 False，否则返回 True。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        prev_position = len(nums) - 1

        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= prev_position:
                prev_position = i

        return prev_position == 0
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次数组
- 空间复杂度：O(1)，使用常数个变量

#### [45. Jump Game II](https://leetcode.com/problems/jump-game-ii/)

这道题的描述是：给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。你的目标是使用最少的跳跃次数到达最后一个位置。

Test cases:

```text
Input: nums = [2,3,1,1,4]
Output: 2

Input: nums = [2,3,0,1,4]
Output: 2
```

这道题的解法是：贪心算法。遍历数组，记录当前能到达的最远位置，如果当前位置大于最远位置，则更新最远位置，同时步数加一。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        res = 0
        cur_far = 0
        cur_end = 0

        for i in range(len(nums) - 1):
            cur_far = max(cur_far, i + nums[i])

            if i == cur_end:
                res += 1
                cur_end = cur_far
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次数组
- 空间复杂度：O(1)，使用常数个变量

### 转换类题

#### [13. Roman to Integer](https://leetcode.com/problems/roman-to-integer/)

这道题的描述是：罗马数字包含以下七种字符：I，V，X，L，C，D 和 M。给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

Test cases:

```text
Input: "III"
Output: 3

Input: "IV"
Output: 4

Input: "IX"
Output: 9

Input: "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.

Input: "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
```

这道题的解法是：遍历字符串，如果当前字符代表的值小于下一个字符代表的值，则减去当前字符代表的值，否则加上当前字符代表的值。

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        res = 0
        roman_map = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        for i in range(len(s) - 1):
            if roman_map[s[i]] < roman_map[s[i + 1]]:
                res -= roman_map[s[i]]
            else:
                res += roman_map[s[i]]

        return res + roman_map[s[-1]]
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次字符串
- 空间复杂度：O(1)，使用常数个变量

#### [12. Integer to Roman](https://leetcode.com/problems/integer-to-roman/)

这道题的描述是：罗马数字包含以下七种字符：I，V，X，L，C，D 和 M。给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内。

Test cases:

```text
Input: 3
Output: "III"

Input: 4
Output: "IV"

Input: 9
Output: "IX"

Input: 58
Output: "LVIII"

Input: 1994
Output: "MCMXCIV"
```

这道题的解法是：遍历罗马数字，如果当前数字小于等于当前罗马数字，则减去当前罗马数字，否则加上当前罗马数字。

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        res = ''
        roman_map = {
            1000: 'M',
            900: 'CM',
            500: 'D',
            400: 'CD',
            100: 'C',
            90: 'XC',
            50: 'L',
            40: 'XL',
            10: 'X',
            9: 'IX',
            5: 'V',
            4: 'IV',
            1: 'I'
        }

        for key in roman_map:
            while num >= key:
                res += roman_map[key]
                num -= key

        return res
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次罗马数字
- 空间复杂度：O(1)，使用常数个变量

#### [273. Integer to English Words](https://leetcode.com/problems/integer-to-english-words/)

这道题的描述是：将非负整数转换为其对应的英文表示。可以保证给定输入小于 2^31 - 1

Test cases:

```text
Input: 123
Output: "One Hundred Twenty Three"

Input: 12345
Output: "Twelve Thousand Three Hundred Forty Five"

Input: 1234567
Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"

Input: 1234567891
Output: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"
```

这道题的解法是：将数字按照三位一组分割，然后将每一组转换为英文表示，最后将每一组的英文表示拼接起来。

```python
class Solution:
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """

        to19 = 'x One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve ' \
           'Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split()

        tens = 'x x Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()

        def words(n):
            if n <= 0:
                return []

            if n < 20:
                return [to19[n]]

            if n < 100:
                return [tens[n // 10]] + words(n % 10)

            if n < 1000:
                return [to19[n // 100]] + ['Hundred'] + words(n % 100)

            if n < 1000 ** 2:
                return words(n // 1000) + ["Thousand"] + words(n % 1000)

            if n < 1000 ** 3:
                return words(n // 1000 ** 2) + ["Million"] + words(n % 1000 ** 2)

            if n < 1000 ** 4:
                return words(n // 1000 ** 3) + ["Billion"] + words(n % 1000 ** 3)

        return " ".join(words(num)) or "Zero"
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次数字
- 空间复杂度：O(1)，使用常数个变量

### Interval

#### [228. Summary Ranges](https://leetcode.com/problems/summary-ranges/)

这道题的描述是：给定一个无重复元素的有序整数数组，返回数组区间范围的汇总。

Test cases:

```text
Input: [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]

Input: [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
```

这道题的解法是：遍历数组，如果当前元素和下一个元素连续，则继续遍历，否则将当前区间加入结果。

首先是一个笨方法：

```python
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        if not nums:
            return nums

        res = []
        start, end = 0, 0
        for i in range(len(nums)):
            if i > 0:
                if nums[i] == nums[i - 1] + 1:
                    end += 1
                else:
                    if start < end:
                        res.append(str(nums[start]) + "->" + str(nums[end]))
                    elif start == end:
                        res.append(str(nums[start]))
                    start = i
                    end = i

        if start < end:
            res.append(str(nums[start]) + "->" + str(nums[end]))
        elif start == end:
            res.append(str(nums[start]))

        return res
```

显然你发现最后还要再加一次，所以我可以用 while 循环来代替 for 循环：

```python
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        res = []
        i = 0

        while i < len(nums):
            j = i

            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1

            if j == i:
                res.append(str(nums[i]))
            else:
                res.append(str(nums[i]) + '->' + str(nums[j]))

            i = j + 1

        return res
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次数组
- 空间复杂度：O(1)，使用常数个变量

#### [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)

这道题的描述是：给定一个区间的集合，请合并所有重叠的区间。

Test cases:

```text
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]

Input: [[1,4],[4,5]]
Output: [[1,5]]
```

这道题的解法是：先将区间按照左端点排序，然后遍历区间，如果当前区间的左端点小于等于上一个区间的右端点，则合并区间，否则将当前区间加入结果。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return intervals

        intervals.sort(key=lambda x: x[0])
        res = [intervals[0]]

        for i in range(1, len(intervals)):
            if intervals[i][0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                res.append(intervals[i])

        return res
```

复杂度分析：

- 时间复杂度：O(nlogn)，排序需要 O(nlogn)，遍历一次数组需要 O(n)
- 空间复杂度：O(1)，使用常数个变量

#### [57. Insert Interval](https://leetcode.com/problems/insert-interval/)

这道题的描述是：给定一个无重叠的有序区间列表，插入一个新的区间，确保列表仍然有序且不重叠。

Test cases:

```text
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
```

这道题的解法是：先将新区间插入到区间列表中，然后按照上一题的方法合并区间。

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval)
        intervals.sort(key=lambda x: x[0])
        res = [intervals[0]]

        for i in range(1, len(intervals)):
            if intervals[i][0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                res.append(intervals[i])

        return res
```

当然这样需要 sort，也可以不 sort，直接遍历区间列表，如果当前区间的右端点小于新区间的左端点，则将当前区间加入结果，如果当前区间的左端点大于新区间的右端点，则将新区间加入结果，否则合并区间。

```python
class Solution:
    def init_new_intervals(self, intervals, newInterval):
        is_inserted = False
        for i in range(len(intervals)):
            if intervals[i][0] <= newInterval[0]:
                inserted_intervals = intervals[:i+1] + [newInterval] + intervals[i+1:]
                is_inserted = True
        # easy to be forgotten
        if not is_inserted:
            inserted_intervals = [newInterval] + intervals
        return inserted_intervals

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        inserted_intervals = self.init_new_intervals(intervals, newInterval)
        res = [inserted_intervals[0]]

        for item in inserted_intervals[1:]:
            if item[0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1], item[1])
            else:
                res.append(item)

        return res
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次区间列表
- 空间复杂度：O(n)，需要一个新的区间列表

#### [986. Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)

这道题的描述是：给定两个有序区间列表，返回两个区间列表的交集。

Test cases:

```text
Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]

Input: A = [[1,7]], B = [[3,10]]
Output: [[3,7]]
```

这道题的解法是：使用双指针，分别指向两个区间列表，如果两个区间有交集，则将交集加入结果，否则将左端点小的区间右移。

```python
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        if not firstList or not secondList:
            return []

        first = 0
        second = 0

        res = []

        while first < len(firstList) and second < len(secondList):
            if firstList[first][0] <= secondList[second][0] and firstList[first][1] >= secondList[second][0]:
                if firstList[first][1] <= secondList[second][1]:
                    res.append([secondList[second][0], firstList[first][1]])
                    first += 1
                else:
                    res.append([secondList[second][0], secondList[second][1]])
                    second += 1

            elif firstList[first][0] >= secondList[second][0] and firstList[first][0] <= secondList[second][1]:
                if firstList[first][1] <= secondList[second][1]:
                    res.append([firstList[first][0], firstList[first][1]])
                    first += 1
                else:
                    res.append([firstList[first][0], secondList[second][1]])
                    second += 1
            else:
                if firstList[first][0] >= secondList[second][0]:
                    second += 1
                else:
                    first += 1

        return res
```

当然我们可以使用更简洁的写法，将上面的 if-else 语句简化一下，就形成了这样的答案：

```python
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        if not firstList or not secondList:
            return []

        first = 0
        second = 0

        res = []

        while first < len(firstList) and second < len(secondList):
            low = max(firstList[first][0], secondList[second][0])
            high = min(firstList[first][1], secondList[second][1])

            if low <= high:
                res.append([low, high])

            if firstList[first][1] < secondList[second][1]:
                first += 1
            else:
                second += 1

        return res
```

复杂度分析：

- 时间复杂度：O(n)，遍历一次区间列表
- 空间复杂度：O(1)，使用常数个变量

#### [252. Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)

这道题的描述是：给定一个区间列表，判断是否存在重叠区间。

Test cases:

```text
Input: [[0,30],[5,10],[15,20]]
Output: false

Input: [[7,10],[2,4]]
Output: true
```

这道题的解法是：先将区间列表按照左端点排序，然后遍历区间列表，如果当前区间的左端点小于上一个区间的右端点，则存在重叠区间。

```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        if not intervals:
            return True

        intervals.sort(key=lambda x: x[0])

        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i-1][1]:
                return False

        return True
```

复杂度分析：

- 时间复杂度：O(nlogn)，排序需要 O(nlogn)的时间复杂度，遍历区间列表需要 O(n)的时间复杂度
- 空间复杂度：O(1)，使用常数个变量

#### [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

这道题的描述是：给定一个区间列表，求出需要的会议室的最小数量。

Test cases:

```text
Input: [[0, 30],[5, 10],[15, 20]]
Output: 2

Input: [[7,10],[2,4]]
Output: 1
```

这道题的解法是：先将区间列表按照左端点排序，然后使用一个最小堆，将第一个区间的右端点加入堆中，然后遍历区间列表，如果当前区间的左端点大于等于堆顶元素，则将堆顶元素弹出，将当前区间的右端点加入堆中，否则将当前区间的右端点加入堆中。最后返回堆的大小。

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

        intervals.sort()

        pq = [intervals[0][1]]

        for i in range(1, len(intervals)):
            if intervals[i][0] >= pq[0]:
                heapq.heappop(pq)
            heapq.heappush(pq, intervals[i][1])

        return len(pq)
```

复杂度分析：

- 时间复杂度：O(nlogn)，排序需要 O(nlogn)的时间复杂度，遍历区间列表需要 O(n)的时间复杂度，每次堆操作需要 O(logn)的时间复杂度
- 空间复杂度：O(n)，使用了一个大小为 n 的最小堆

#### [452. Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

这道题的描述是：给定一个区间列表，求出最少需要多少箭可以将所有气球都戳破。

Test cases:

```text
Input: [[10,16], [2,8], [1,6], [7,12]]
Output: 2

Input: [[1,2],[3,4],[5,6],[7,8]]
Output: 4
```

这道题的解法是：先将区间列表按照右端点排序，然后遍历区间列表，如果当前区间的左端点大于上一个区间的右端点，则需要增加一支箭。

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0

        points.sort(key = lambda x: x[1])

        arrows = 1
        first_end = points[0][1]

        for x_start, x_end in points:
            if first_end < x_start:
                arrows += 1
                first_end = x_end

        return arrows
```

复杂度分析：

- 时间复杂度：O(nlogn)，排序需要 O(nlogn)的时间复杂度，遍历区间列表需要 O(n)的时间复杂度
- 空间复杂度：O(1)，使用常数个变量

#### [435. Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)

这道题的描述是：给定一个区间列表，求出最少需要移除多少区间可以使得剩下的区间不重叠。

Test cases:

```text
Input: [[1,2],[2,3],[3,4],[1,3]]
Output: 1

Input: [[1,2],[1,2],[1,2]]
Output: 2
```

这道题的解法是：先将区间列表按照右端点排序，然后使用一个变量记录当前区间的右端点，遍历区间列表，如果当前区间的左端点小于等于当前区间的右端点，则需要移除当前区间，否则更新当前区间的右端点为当前区间的右端点。最后返回移除的区间数量。

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

        intervals.sort(key=lambda x: x[1])

        end = intervals[0][1]
        res = 0

        for i in range(1, len(intervals)):
            if intervals[i][0] < end:
                res += 1
            else:
                end = intervals[i][1]

        return res
```

复杂度分析：

- 时间复杂度：O(nlogn)，排序需要 O(nlogn)的时间复杂度，遍历区间列表需要 O(n)的时间复杂度
- 空间复杂度：O(1)，使用常数个变量
