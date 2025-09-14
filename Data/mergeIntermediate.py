import pandas as pd

def merge_csv(file1, file2, output_file):
    """
    Merge two CSV files on the DOI column.
    Keeps DOI, Layers from file1 and all other columns from file2.
    """
    # Load both CSVs
    df1 = pd.read_csv(file1)   # has DOI, Layers
    df2 = pd.read_csv(file2)   # has DOI and other data

    # Keep only DOI and Layers from df1
    df1_subset = df1[['DOI', 'Layer']]

    # Merge
    merged = pd.merge(df2, df1_subset, on="DOI", how="left")

    # Save output
    merged.to_csv(output_file, index=False)
    print(f"Merged file saved as: {output_file}")


if __name__ == "__main__":
    file1 = "citation_scores_10.1038_nature14236.csv"         # has DOI, Layers
    file2 = "doiData.csv"         # has DOI, other data
    output_file = "doiDataLayers.csv"

    merge_csv(file1, file2, output_file)
