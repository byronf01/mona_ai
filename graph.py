import re
import math
import matplotlib.pyplot as plt

x = []
y = []

with open('graph_data.txt', 'r') as f:
    for line in f:
        data = re.findall(r'Epoch (\d+): (\d+.\d+)', line)
        data = data[0]
        x.append(int(data[0]))
        y.append(float(data[1]))

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train loss Transformer model')
plt.xticks(range(min(x) - 1, max(x), 200))  
plt.yticks(range(math.ceil(min(y)) - 1, int(max(y)) + 1, 2))
plt.show()