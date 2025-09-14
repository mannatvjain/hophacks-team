import pandas as pd

def parse_doi_array(array_str):
    """
    Convert a string representing an array of DOIs into a Python list.
    Handles cases like:
    "[10.1001/abc, 10.1002/xyz]"  -> ['10.1001/abc', '10.1002/xyz']
    "['10.1003/def','10.1004/ghi']" -> ['10.1003/def', '10.1004/ghi']
    """
    # Remove brackets if present
    array_str = array_str.strip()
    if array_str.startswith("[") and array_str.endswith("]"):
        array_str = array_str[1:-1]

    # Split by comma and remove quotes/whitespace
    items = [x.strip().strip("'").strip('"') for x in array_str.split(',') if x.strip()]
    return items

def split_dois(main_csv, doi_arrays_csv, output_common_csv, output_only_in_arrays_csv):
    # Load main CSV
    main_df = pd.read_csv(main_csv)
    main_dois = set(main_df['DOI'])
    print(f"DOIs in main CSV: {len(main_dois)}")

    # Load DOI arrays CSV
    arrays_df = pd.read_csv(doi_arrays_csv, header=None, names=['DOI_array'])

    # Flatten DOI arrays into a single set
    all_array_dois = set()
    array_rows = []
    for idx, row in arrays_df.iterrows():
        doi_list = parse_doi_array(row['DOI_array'])
        all_array_dois.update(doi_list)
        array_rows.append(doi_list)
    print(f"Total unique DOIs in arrays CSV: {len(all_array_dois)}")

    # DOIs present in both
    common_dois = main_dois.intersection(all_array_dois)
    common_df = main_df[main_df['DOI'].isin(common_dois)]
    common_df.to_csv(output_common_csv, index=False)
    print(f"Saved common DOIs to {output_common_csv} ({len(common_df)} rows)")

    # DOIs only in arrays CSV
    only_in_arrays = all_array_dois - main_dois

    # Reconstruct rows from original arrays CSV but keep only DOIs missing in main CSV
    rows_only_in_arrays = []
    for doi_list in array_rows:
        filtered_list = [doi for doi in doi_list if doi in only_in_arrays]
        if filtered_list:
            rows_only_in_arrays.append(filtered_list)

    # Save to CSV
    only_in_arrays_df = pd.DataFrame({'DOI_array': rows_only_in_arrays})
    only_in_arrays_df.to_csv(output_only_in_arrays_csv, index=False)
    print(f"Saved DOIs only in arrays CSV to {output_only_in_arrays_csv} ({len(only_in_arrays_df)} rows)")

if __name__ == "__main__":
    main_csv = "final_merged_with_abstracts_filtered.csv"
    doi_arrays_csv = "top10_paths_to_source_arrays.csv"
    output_common_csv = "top10_wAbstracts.csv"
    output_only_in_arrays_csv = "top10_woAbstracts.csv"

    split_dois(main_csv, doi_arrays_csv, output_common_csv, output_only_in_arrays_csv)
