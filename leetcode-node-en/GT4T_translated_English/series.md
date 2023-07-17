## Type questions

There are many typical series of questions, such as buying and selling stocks series I, II, III, IV and so on. Here I call them all type problems, because they are all the same kind of problems, and the solutions are similar.

### Buy and sell stocks

#### [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

The description of this problem is: Given an array, its ith element is the price of a given stock on the ith day. If you are only allowed to make one trade at most (buying and selling one stock), design an algorithm to calculate the maximum profit you can make. Note that you can't sell stocks before you buy them.

Test cases:

```text
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
```

The solution to this problem is to traverse the array, record the current minimum value, then calculate the difference between the current value and the minimum value, compare it with the current maximum difference, and update the maximum difference if it is greater than the current maximum difference.

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price, max_profit = prices[0], float('-inf')

        for i in range(len(prices)):
            min_price = min(prices[i], min_price)
            max_profit = max(max_profit, prices[i] - min_price)

        return max_profit
```

Complexity analysis:

- Time complexity: O (n), traverse the array once
- Space complexity: O (1), using a constant number of variables

#### [122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

The description of this problem is: Given an array, its ith element is the price of a given stock on the ith day. Design an algorithm to calculate the maximum profit you can make. You can make as many trades (buy and sell a stock multiple times) as you can.

Test cases:

```text
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
```

The solution to this problem is to traverse the array, and if the current value is greater than the previous value, calculate the difference between the current value and the previous value and add it to the maximum profit.

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0

        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                max_profit += prices[i] - prices[i - 1]

        return max_profit
```

Complexity analysis:

- Time complexity: O (n), traverse the array once
- Space complexity: O (1), using a constant number of variables

#### [123. Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)

The description of this problem is: Given an array, its ith element is the price of a given stock on the ith day. Design an algorithm to calculate the maximum profit you can make. You can complete up to two transactions.

Test cases:

```text
Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
```

The solution to this problem is dynamic programming. Four variables are defined, which represent the maximum profit of the first purchase, the first sale, the second purchase and the second sale. Iterate through the array and update the four variables.

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

This method has an O (N ^ 2) time complexity and will time out. Here's a way to explain it more clearly:

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

Complexity analysis:

- Time complexity: O (n), traverse the array once
- Space complexity: O (1), using a constant number of variables

#### [188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)

The description of this problem is: Given an array, its ith element is the price of a given stock on the ith day. Design an algorithm to calculate the maximum profit you can make. You can complete up to K transactions.

Test cases:

```text
Input: k = 2, prices = [2,4,1]
Output: 2

Input: k = 2, prices = [3,2,6,5,0,3]
Output: 7
```

The solution to this problem is dynamic programming. Define a two-dimensional array. The ith row and the jth column represent the maximum profit of the ith transaction on the jth day. Iterate through the array and update the two-dimensional array.

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

Complexity analysis:

- Time complexity: O (NK), traverse the array once
- Space complexity: O (K), using K variables

### Jump Game

#### [55. Jump Game](https://leetcode.com/problems/jump-game/)

The description of this problem is: Given an array of non-negative integers, you are initially in the first position of the array. Each element in the array represents the maximum length you can jump at that position. Determine whether you can reach the last position.

Test cases:

```text
Input: nums = [2,3,1,1,4]
Output: true

Input: nums = [3,2,1,0,4]
Output: false
```

The solution to this problem is the greedy algorithm. Traverses the array and records the farthest position that can be reached currently. If the current position is greater than the farthest position, it returns False, otherwise it returns True.

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        prev_position = len(nums) - 1

        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= prev_position:
                prev_position = i

        return prev_position == 0
```

Complexity analysis:

- Time complexity: O (n), traverse the array once
- Space complexity: O (1), using a constant number of variables

#### [45. Jump Game II](https://leetcode.com/problems/jump-game-ii/)

The description of this problem is: Given an array of non-negative integers, you are initially in the first position of the array. Each element in the array represents the maximum length you can jump at that position. Your goal is to get to the last position using the fewest number of jumps.

Test cases:

```text
Input: nums = [2,3,1,1,4]
Output: 2

Input: nums = [2,3,0,1,4]
Output: 2
```

The solution to this problem is the greedy algorithm. Traversing the array, recording the farthest position that can be reached at present, if the current position is greater than the farthest position, updating the furthest position, and adding one to the number of steps.

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

Complexity analysis:

- Time complexity: O (n), traverse the array once
- Space complexity: O (1), using a constant number of variables

### Conversion questions

#### [13. Roman to Integer](https://leetcode.com/problems/roman-to-integer/)

The description of this question is: Roman numerals contain the following seven characters: I, V, X, L, C, D and M. Given a Roman numeral, convert it to an integer. Make sure the input is in the range of 1 to 3999.

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

The solution to this problem is to traverse the string, if the value represented by the current character is less than the value represented by the next character, then subtract the value of the current character, otherwise add the value of the current character.

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

Complexity analysis:

- Time complexity: O (n), traverse the string once
- Space complexity: O (1), using a constant number of variables

#### [12. Integer to Roman](https://leetcode.com/problems/integer-to-roman/)

The description of this question is: Roman numerals contain the following seven characters: I, V, X, L, C, D and M. Given an integer, convert it to a Roman numeral. Make sure the input is in the range of 1 to 3999.

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

The solution to this problem is to traverse the Roman numerals, if the current number is less than or equal to the current Roman numeral, then subtract the current Roman numeral, otherwise add the current Rome numeral.

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

Complexity analysis:

- Time complexity: O (n), traversing Roman numerals once
- Space complexity: O (1), using a constant number of variables

#### [273. Integer to English Words](https://leetcode.com/problems/integer-to-english-words/)

The description of this problem is to convert a non-negative integer into its corresponding English representation. A given input is guaranteed to be less than 2 ^ 31-1

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

The solution to this problem is to divide the numbers into groups of three digits, then convert each group into English representations, and finally splice the English representations of each group together.

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

Complexity analysis:

- Time complexity: O (n), traverse the number once
- Space complexity: O (1), using a constant number of variables

### Interval

#### [228. Summary Ranges](https://leetcode.com/problems/summary-ranges/)

The description of this problem is: Given an ordered integer array without repeated elements, return the summary of the range of the array.

Test cases:

```text
Input: [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]

Input: [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
```

The solution to this problem is: traverse the array, if the current element and the next element are consecutive, continue to traverse, otherwise add the current interval to the result.

The first is a stupid method:

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

Obviously, you find that you have to add it again at the end, so I can use the while loop instead of the for loop:

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

Complexity analysis:

- Time complexity: O (n), traverse the array once
- Space complexity: O (1), using a constant number of variables

#### [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)

Given a set of intervals, merge all the overlapping intervals.

Test cases:

```text
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]

Input: [[1,4],[4,5]]
Output: [[1,5]]
```

The solution to this problem is to sort the intervals according to the left endpoint, and then traverse the intervals. If the left endpoint of the current interval is less than or equal to the right endpoint of the previous interval, the intervals will be merged. Otherwise, the current interval will be added to the result.

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

Complexity analysis:

- Time complexity: O (nlogn), sorting requires O (nlogn), traversing the array once requires O (n)
- Space complexity: O (1), using a constant number of variables

#### [57. Insert Interval](https://leetcode.com/problems/insert-interval/)

The description of this problem is: Given a list of ordered intervals without overlap, insert a new interval to ensure that the list is still ordered and does not overlap.

Test cases:

```text
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
```

The solution to this problem is to insert the new interval into the interval list, and then merge the intervals according to the method of the previous problem.

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

Of course, sort is required in this way, or it can be omitted. The interval list is traversed directly. If the right endpoint of the current interval is less than the left endpoint of the new interval, the current interval is added to the result. If the left endpoint of the current interval was greater than the right endpoint of new interval, the new interval is added to the result. Otherwise, the intervals are merged.

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

Complexity analysis:

- Time complexity: O (n), traverse the interval list once
- Space complexity: O (n), requiring a new list of intervals

#### [986. Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)

The description of this problem is: Given two ordered interval lists, return the intersection of the two interval lists.

Test cases:

```text
Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]

Input: A = [[1,7]], B = [[3,10]]
Output: [[3,7]]
```

The solution to this problem is to use double pointers to point to two interval lists respectively. If the two intervals intersect, the intersection will be added to the result. Otherwise, the interval with the smaller left endpoint will be moved to the right.

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

Of course, we can use a more concise way to simplify the if-else statement above to form such an answer:

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

Complexity analysis:

- Time complexity: O (n), traverse the interval list once
- Space complexity: O (1), using a constant number of variables

#### [252. Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)

The description of this problem is: Given a list of intervals, determine whether there are overlapping intervals.

Test cases:

```text
Input: [[0,30],[5,10],[15,20]]
Output: false

Input: [[7,10],[2,4]]
Output: true
```

The solution to this problem is to sort the list of intervals according to the left endpoint, and then traverse the list of intervals. If the left endpoint of the current interval is less than the right endpoint of the previous interval, there are overlapping intervals.

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

Complexity analysis:

- Time complexity: O (nlogn), sorting requires O (nlogn) time complexity, traversing the interval list requires O (n) time complexity
- Space complexity: O (1), using a constant number of variables

#### [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

The description of this problem is: Given a list of intervals, find the minimum number of meeting rooms needed.

Test cases:

```text
Input: [[0, 30],[5, 10],[15, 20]]
Output: 2

Input: [[7,10],[2,4]]
Output: 1
```

The solution to this problem is: first sort the interval list according to the left endpoint, then use a minimum heap, add the right endpoint of the first interval to the heap, and then traverse the interval list. If the left endpoint of the current interval is greater than or equal to the top element of the heap, pop up the top element of the heap and add the right endpoint of current interval to the heap. Otherwise, add the right end point of current interval into the heap. Finally returns the size of the heap.

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

Complexity analysis:

- Time complexity: O (nlogn), sorting requires O (nlogn) time complexity, traversing the interval list requires O (n) time complexity, and each heap operation requires O (logn) time complexity
- Space complexity: O (n), using a minimum heap of size n

#### [452. Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

Given a list of intervals, find the minimum number of arrows needed to burst all the balloons.

Test cases:

```text
Input: [[10,16], [2,8], [1,6], [7,12]]
Output: 2

Input: [[1,2],[3,4],[5,6],[7,8]]
Output: 4
```

The solution to this problem is to sort the list of intervals according to the right endpoint, and then traverse the list of intervals. If the left endpoint of the current interval is greater than the right endpoint of the previous interval, an arrow needs to be added.

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

Complexity analysis:

- Time complexity: O (nlogn), sorting requires O (nlogn) time complexity, traversing the interval list requires O (n) time complexity
- Space complexity: O (1), using a constant number of variables

#### [435. Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)

The problem is described as: Given a list of intervals, find the minimum number of intervals that need to be removed so that the remaining intervals do not overlap.

Test cases:

```text
Input: [[1,2],[2,3],[3,4],[1,3]]
Output: 1

Input: [[1,2],[1,2],[1,2]]
Output: 2
```

The solution of this problem is: first sort the interval list according to the right endpoint, and then use a variable to record the right endpoint of the current interval, traverse the interval list, if the left endpoint of the current interval is less than or equal to the right endpoint of current interval, then you need to remove the current interval.Otherwise, you need to update the right endpoints of current interval as the right endpoints. Finally, return the number of intervals removed.

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

Complexity analysis:

- Time complexity: O (nlogn), sorting requires O (nlogn) time complexity, traversing the interval list requires O (n) time complexity
- Space complexity: O (1), using a constant number of variables
