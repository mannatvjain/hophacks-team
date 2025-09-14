import pandas as pd
import ast

# Load CSVs
paths_file = "top10_paths_to_source_arrays.csv"
summary_file = "summary.csv"

paths_df = pd.read_csv(paths_file)
summary_df = pd.read_csv(summary_file)

mapped_data = []

for idx, row in paths_df.iterrows():
    path_str = str(row[0])
    
    try:
        # Convert the string representation of the list to an actual list
        doi_array = ast.literal_eval(path_str)
        last_doi = doi_array[-1] if len(doi_array) > 0 else None
    except Exception as e:
        print(f"Skipping row {idx}, could not parse DOIs: {path_str} ({e})")
        last_doi = None

    # Get the corresponding summary
    summary_text = summary_df.iloc[idx]["Summary"] if idx < len(summary_df) else None

    mapped_data.append({
        "DOI": last_doi,
        "Summary": summary_text
    })

# Save the result
mapped_df = pd.DataFrame(mapped_data)
mapped_df.to_csv("mappedSummary.csv", index=False)

print("Saved mappedSummary.csv")
