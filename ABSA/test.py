M = [
    [0.61, 0.33, 0.87], 
    [0.93, 0.71, 0.87],
    [1.32, 1.09, 0.87], 
    [1.71, 1.47, 0.87],
    [2.08, 1.87, 0.87],
    [2.48, 2.26, 0.87],
    [2.89, 2.65, 0.87],
    [3.29, 3.04, 0.87], 
    [3.68, 3.44, 0.87]
]

for line in M:
    print(1.2*sum(line)/len(line))