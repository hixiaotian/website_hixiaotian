<hr>

###Tricks in Bit Operation

Bit operations contain the following:
&, |, ^, ~, <<, >>, !

In the above operation, I love shift and XOR operations. Why?

Let's see what can they do!

####Tricks:

##### 0. Obtain useful masks!

Mask is quite crucial in bit computation. This is set to 0 since all the bit operation always depend on MASKS!

Here are a list of masks that is quite useful:
(I use int type here)
```python
# a) obtain FFFFFFFF
    mask_FFFFFFFF = ~0
# b) obtain 0000FFFF
    mask_0000FFFF = 1 << 16 - 1
# c) obtain FFFF0000
    mask_0000FFFF = ~0 << 16
# d) obtain 55555555
    mask_55555555 = "Not Available now"
```

##### 1.  left shift << can be used as Power-of-2 Multiply:

```python
# gives u * 2^3
u << 3
# gives u * 24
(u << 5) - (u << 3)
```

Remember that this is also good when u is a negative number. (But not the case on the boundary of type)
such as: 
```python
# good case
u = -1  # u = 0xFFFFFFFF --- u = -1
u <<= 1 # u = 0xFFFFFFFE --- u = -2
u <<= 1 # u = 0xFFFFFFFC --- u = -4
```

Consider the bad case as always!
```python
#bad case
u = 2147483648 # u = 0x7FFFFFFF
u << 1 # u = 0xFFFFFFFF

v = -2147483648 # v = 0x10000000
v << 1 # v = 0x00000000
```

##### 2. Right shift can be used as Power-of-2 divide with rounding
In the right shift, it is not always rational divided due to rounding, it is quite similar to ``5/2 = 2`` in C.

Notice that we should consider positive and negative cases separately.

For positive cases:
![--2021-09-14---6.44.56](http://hixiaotian.com/content/images/2021/09/--2021-09-14---6.44.56.png)

For negative cases:
![--2021-09-14---6.45.09](http://hixiaotian.com/content/images/2021/09/--2021-09-14---6.45.09.png)

It is wrong, because in negative cases, we should round up rather round down! So we need to change the right shift code to:
```python
# this denote for if x is a negative number of divide 2 with rounding
x = -5
x = x + (1 << k) - 1) >> k
# In this way, it is similar to x >> k in positive number
```


##### 3. Eliminate / Obtain the last 1 in binary

Eliminate the last 1 in binary
n & (n - 1)

```python
01011101 &
01011100
--------
01011100
```

Obtain the last 1 in binary
n & (-n)
```python
10110100 &
01001100
--------
00000100
```

##### 4. XOR: My favorite operator
```python
x ^ 0 = x
x ^ x = 0
x ^ -1 = ~x

```

#### Leetcode problems:

##### 461 (easy). Hamming Distance
Description: get the numbers of bit differences in binary
```
Input: x = 1, y = 4
Output: 2
Explanation: 
1   (0 0 0 1)
4   (0 1 0 0)
```

Solution 1 with O(n):
```python
def hammingDistance(self, x: int, y: int) -> int:
    temp = x ^ y #get the diff numbers
    count = 0
    while temp != 0:
        temp &= temp - 1 #eliminate 1 at a time
        count += 1
    return count
```

Solution 2 with special function in O(n):
```python
def hammingDistance(self, x: int, y: int) -> int:
    bin(x ^ y).count('1')
```

Solution 3 (Fastest!) with shift in O(n):
```python
def hammingDistance(self, x, y):
        res = x ^ y
        counter = 0
        for i in range(32):
            if res >> i & 1 == 1: #using shift is much faster!!!!!
                counter += 1
        return counter
```


##### 136 (easy). Single Number
Description: find the number in the list that is single
```
Input: [4,1,2,1,2]
Output: 4
```

Solution 1 with O(n):

Consider that:
```
x ^ x = 0
x ^ 0 = x
```
Then use loooooop for entire list, the operation would be:
```
4 ^ 1 ^ 2 ^ 1 ^ 2 => 4
```
Thus, the solution would be:

```python
def singleNumber(self, nums: List[int]) -> int:
    res = 0
    for item in nums: 
        res ^= item
    return res
```

##### 268 (easy). Find missing number

Input: [3,0,1]
Output: 2

Solution 1 use XOR in O(1):
```python
def missingNumber(self, nums: List[int]) -> int:
    ret = 0
    for i in range(len(nums)):
        ret ^= i ^ nums[i]
    return ret ^ len(nums) # I love this one!
```
Why?
Consider that:
``ret = (0 ^ 3) ^ (1 ^ 0) ^ (2 ^ 1)``
and return ``len(nums)`` if ``ret = 0``
    return ``ret`` if ``ret != 0``


Solution 2 use enumerate in O(1):
```python
def missingNumber(self, nums: List[int]) -> int:
    nums.sort()
    for index, item in enumerate(nums):
        if index != item:
            return index
    else: # else is a good usage for "for"
        return nums[-1] + 1
```


##### 260 (Medium). Single Number III

Solution 1: that is <b>not</b> using bit operation, but I like it pretty much, since it use the set() to eliminate duplicates, btw, this solution is also suitable for the above problem 136:
```python
def singleNumber(self, nums: List[int]) -> List[int]:
    res = set()
    for item in nums:
        if item in res: res.remove(item)
        else: res.add(item)
    return res
```

Solution 2: bit operation solution:
```python

# The bitmask will have all the differences between x and y (the two singles).
# x has 0 and y has 1 or vice-versa in those bits.
# You take one difference, the rightmost one.

# Iterate through array again dividing into two. 
# The ones which has 1 on the rightmost different bit into one group and those who have 0 into another group.
# Group1 : [x^a^a^b^b] = x
# Group2 : [y^c^c^d^d] = y
# bitmask^x = y;

def singleNumber(self, nums: List[int]) -> List[int]:
        bitmask = 0
        for num in nums:
            bitmask ^= num    
        diff = bitmask & (-bitmask)
        x = 0
        for num in nums:
            if num & diff:
                x ^= num    
        return [x, bitmask ^ x]
```

##### 190. Reverse Bits

Solution 1: again, here I would like to reverse the bits by using the smart reverse method in python:

```python 
def reverseBits(self, n: int) -> int:
    return int('{:032b}'.format(n)[::-1], 2)
```
``{:032b}'.format(n)`` is for revert int value to string with 32 bit binary
``[::-1]`` is the smart way to reverse a string in Python
``int(x, 2)`` is to output the int value as a binary number

Solution 2:
```python
def reverseBits(self, n: int) -> int:
    ret, power = 0, 31
    while n:
        ret += (n & 1) << power
        n >>= 1
        power -= 1
    return ret
```

##### 231. Power of 2:

Solution: use eliminate last 1 method:
KEY POINT: Remember to consider 0!!!
```python
def isPowerOfTwo(self, n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
```

##### 342. Power of 4:
Solution:
```python
def isPowerOfFour(self, n: int) -> bool:
    if(n<0):
        return False
    b = bin(n)
    if(b.count("1") == 1 and (b[3:].count("0") % 2 == 0)):
        return True
    return False
```

##### 693. Binary Number with Alternating Bits

Solution 1:
```python
def hasAlternatingBits(self, n: int) -> bool:
    return n & (n >> 1) == 0 and n & (n >> 2) == n >> 2
```

Solution 2:
```python
def hasAlternatingBits(self, n: int) -> bool:
    #      10101010101
    #  +    1010101010    ( number >> 1 )
    #  ---------------
    #  =   11111111111
    #  &  100000000000
    #  ---------------
    #  =             0    ( power of two )
    n = n ^ (n >> 1)
    return n & n + 1 == 0
```