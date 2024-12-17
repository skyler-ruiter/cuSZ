import numpy as np

def read_f32_file(file_path):
  with open(file_path, 'rb') as f:
    data = np.fromfile(f, dtype=np.float32)
  return data

file_path = '../data/280953867/vx.f32'
data = read_f32_file(file_path)

# print the first 10 elements
print(data[:10])

# print the total number of elements
print(len(data))