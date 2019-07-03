from itertools import combinations

a = [1, 2, 3, 4, 5, 6]
result = [list(i) for i in combinations(a, 3)]
print(result)
print(len(result))