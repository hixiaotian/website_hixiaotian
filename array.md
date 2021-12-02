Here we first talk about one dimensional array: which is itself, array.
Then we are going to introduce the two dimensional array: Metrics.  

#### Array:
As always we go with tricks in arrays first!

#### Array Tricks!
1. initialization
```python
array = [1, 3, 4]
# use * to initialize
l = [0] * len(array)
# use list generator
l = [0 for _ in len(array)]
```

2. From behind
```python
array = [1, 2, 3]
lastOne = array[-1]
lastTwo = array[-2:]
lastThree = array[-3:]
```

3. Copy array
```python 
array = [1, 2, 3]
# WRONG!!!!
c = array

# shadow copy: can only copy array, cannot copy nested array!!
# CORRECT COPY
c = array[:]
# or
c = array.copy()

# deep copy: can copy whatever you want
import copy
c = copy.deepcopy(array)
```

4. Enumerate
```python
array = [1, 2, 3]

for index, item in enumerate(array):
    print("index:", index)
    print("item:", item)
# Result:
# index: 0
# item 1
# index: 1
# item 2
# index: 2
# item 3
```

5. zip
```python
x = [1, 2, 3]
y = [4, 5, 6]
zipped = zip(x, y)
# zipped = [(1, 4), (2, 5), (3, 6)
```

6. sort
```python
l1 = [(1,2), (0,1), (3,10)]
l2 = l1[:]

l2.sort()
# l2 = [(0, 1), (1, 2), (3, 10)]
l2 = sorted(l1)
# L2 = [(0, 1), (1, 2), (3, 10)]

l2.sort(reverse=True)
# l2 = [(3, 10), (1, 2), (0, 1)]
l2 = sorted(l1, reverse=True)
# l2 = [(3, 10), (1, 2), (0, 1)]

l2.sort(key=lambda x: x[1])
# l2 = [(0, 1), (1, 2), (3, 10)]
l2 = sorted(l1, key=lambda x: x[1])
# l2 = [(0, 1), (1, 2), (3, 10)]

l2.sort(key=lambda x: x[1], reverse=1)
# l2 = [(3, 10), (1, 2), (0, 1)]
l2 = sorted(l1, key=lambda x: x[1], reverse=1)
# l2 = [(3, 10), (1, 2), (0, 1)]
```

7. conversion
```python
array = [1, 1, 2, 3]
# convert to set
b = set(array)
# -> b = {1, 2, 3}

# convert to string
# cannot directly convert
str_array = []
for item in array:
    str_array.append(str(item))
b = ''.join(str_array)
# b = "1123"
```

#### Leetcode with Array:

##### 238. Move Zeroes

Description:

Solution 1 with exchange:
```python
def moveZeroes(self, nums: List[int]) -> None:
    pos = 0
    for i in range(len(nums)):
        ele = nums[i]
        if ele != 0:
            nums[pos], nums[i] = nums[i], nums[pos]
            pos += 1
```



##### 566.Reshape the Matrix

Description:
Solution 1 with converting to one line then convert to others:
```python
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        rows = len(mat)
        cols = len(mat[0])     
        if r * c != rows * cols:
            return mat
        res = [mat[row][col] for row in range(rows) for col in range(cols)]

        ret = []
        for i in range(0, len(res), c):
            ret.append(res[i:i + c])
            
        return ret
```

Solution 2 with / and % (my god, 0ms):
```java
public int[][] matrixReshape(int[][] nums, int r, int c) {
    int m = nums.length, n = nums[0].length;
    if (m * n != r * c) {
        return nums;
    }
    int[][] reshapedNums = new int[r][c];
    int index = 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            reshapedNums[i][j] = nums[index / n][index % n];
            index++;
        }
    }
    return reshapedNums;
}
```

##### 485. Max consecutive 1
```python
def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        maxn, cur = 0, 0
        for item in nums:
            cur = 0 if item == 0 else cur + 1
            maxn = max(maxn, cur)
        return maxn
```

##### 240. Search a 2D Matrix II
```Python
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    rows = len(matrix)
    cols = len(matrix[0])
    row, col = rows - 1, 0
    while col < cols and row >= 0:
        if matrix[row][col] > target:
            row -= 1
        elif matrix[row][col] < target:
            col += 1
        else:
            return 1
    return 0
```