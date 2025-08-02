import os
from collections import Counter

data_dir = "data"
print("Class distribution:")
print(Counter([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]))