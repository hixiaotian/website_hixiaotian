
##### 1143. Longest Common Subsequence

Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.

```python
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    rows, cols = len(text1) + 1, len(text2) + 1
    grid = [[0] * cols for _ in range(rows)]
    
    for n in range(1, cols):
        for m in range(1, rows):
            if text2[n - 1] == text1[m - 1]:
                grid[m][n] = 1 + grid[m - 1][n - 1]
            else:
                grid[m][n] = max(grid[m - 1][n], grid[m][n - 1])
    print(grid)   
    return grid[rows - 1][cols - 1]
```