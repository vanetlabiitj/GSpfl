file_path = "../store_results/evaluate_client_results.txt"
# Open the file for writing
with open(file_path, "r") as file:  # Open in text mode
    res = file.read()  # Read the entire file

# Split data into rows
rows = res.strip().split('\n')

# Initialize variables to store sums
macro_s = 0
micro_s = 0
macro_m = 0
micro_m = 0
macro_d = 0
micro_d = 0

# Iterate over rows and calculate sums
for row in rows:
    cols = row.split(',')
    cid = float(cols[0])  # Second-last column
    if cid in [0, 1, 2]:
      micro_d += float(cols[-2])
      macro_d += float(cols[-1])
    elif cid in [3, 4, 5, 6]:

      micro_m += float(cols[-2])
      macro_m += float(cols[-1])
    else:
      micro_s += float(cols[-2])
      macro_s += float(cols[-1])

print(f"Dense macro and micro: {(macro_d)/3}, {micro_d/3}")
print(f"Moderate macro and micro: {macro_m/4}, {micro_m/4}")
print(f"Sparse macro and micro: {macro_s/3}, {micro_s/3}")
