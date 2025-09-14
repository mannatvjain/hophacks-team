import pandas as pd

# Load the CSV
input_csv = "final_merged_with_abstracts_filtered.csv"
df = pd.read_csv(input_csv)

# Sort by 'Score' in descending order
df_sorted = df.sort_values(by="Score", ascending=False)

# Save to a new CSV
output_csv = "abstracts_filtered_score.csv"
df_sorted.to_csv(output_csv, index=False)

print(f"Saved sorted CSV to {output_csv}")
