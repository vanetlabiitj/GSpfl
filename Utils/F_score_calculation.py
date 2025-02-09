import re
import pandas as pd

def readfile():
    file_path = "../store_results/class_wise_report.txt"
    # Open the file and read its contents
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines

def convert_df(lines):
    # List to store results as tuples (cid, class_id, precision, recall, f1_score)
    data = []
    # Regular expression pattern for class-wise metrics
    pattern = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    # Regular expression for extracting cid (client ID)
    cid_pattern = re.compile(r"^\s*(\d+),")
    # Initialize a variable to track the current client ID (cid)
    current_cid = None
    # Iterate through each line to extract data
    for line in lines:
        # Match client ID (e.g., 2, at the start of the report)
        cid_match = cid_pattern.match(line.strip())
        if cid_match:
            current_cid = int(cid_match.group(1))  # Extract cid
        # Match class-wise metrics after client ID
        elif pattern.match(line):
             match = pattern.match(line)
             class_id = int(match.group(1))  # Class ID
             precision = float(match.group(2))  # Precision
             recall = float(match.group(3))  # Recall
             f1_score = float(match.group(4))  # F1-score
             # Add the data as a tuple
             if current_cid is not None:
                   data.append((current_cid, class_id, precision, recall, f1_score))

    # Convert the data list into a pandas DataFrame
    df = pd.DataFrame(data, columns=["cid", "class_id", "precision", "recall", "f1_score"])
    return df


file = readfile()
df =convert_df(file)
#print(df.head())

class_avg = df.groupby('class_id')[['precision', 'recall', 'f1_score']].mean()

# Resetting the index to make 'class_id' a column again
class_avg = class_avg.reset_index()

# Print the resulting DataFrame
print(class_avg)



