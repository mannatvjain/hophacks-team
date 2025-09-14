import pandas as pd

def merge_datasets(doi_names_file, citation_scores_file, edges_file, output_file):
    """
    Merge doiNames.csv + citation_scores first, then merge with edges_output.csv.
    Missing References are filled with empty array.
    Final columns: DOI, References, Score, Title, Authors, Year
    """
    # Load files
    doi_df = pd.read_csv(doi_names_file)              # DOI, Title, Authors, Year
    scores_df = pd.read_csv(citation_scores_file)    # DOI, Score
    edges_df = pd.read_csv(edges_file)               # DOI, References

    # Step 1: Merge doiNames + citation_scores on DOI
    combined = pd.merge(doi_df, scores_df, on='DOI', how='inner')
    print(f"Combined doiNames + citation_scores: {len(combined)} rows")

    # Step 2: Merge with edges_output.csv (left join, keep all DOIs from combined)
    merged = pd.merge(combined, edges_df[['DOI', 'References']], on='DOI', how='left')

    # Fill missing References with empty array
    merged['References'] = merged['References'].apply(lambda x: x if pd.notna(x) else '[]')

    # Reorder columns: DOI, References, Score, Title, Authors, Year
    merged = merged[['DOI', 'References', 'Score', 'Title', 'Authors', 'Year']]

    # Save output
    merged.to_csv(output_file, index=False)
    print(f"Final merged CSV saved as: {output_file}, total rows: {len(merged)}")


if __name__ == "__main__":
    doi_names_file = "doiNames.csv"
    citation_scores_file = "citation_scores_10.1038_nature14236.csv"
    edges_file = "edges_output.csv"
    output_file = "final_merged.csv"

    merge_datasets(doi_names_file, citation_scores_file, edges_file, output_file)
