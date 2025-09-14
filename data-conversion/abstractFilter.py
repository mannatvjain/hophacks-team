import pandas as pd

def parse_doi_array(array_str):
    """
    Safely parse a string representing an array of DOIs into a list of DOI strings.
    Ignores rows that are not DOI arrays like ['Path_to_Source'].
    """
    if not isinstance(array_str, str):
        return []

    array_str = array_str.strip()

    # Remove surrounding double quotes if present
    if array_str.startswith('"') and array_str.endswith('"'):
        array_str = array_str[1:-1].strip()

    # Ignore non-DOI rows
    if 'Path_to_Source' in array_str:
        return []

    # Remove brackets
    if array_str.startswith('[') and array_str.endswith(']'):
        array_str = array_str[1:-1]

    # Split by comma, strip whitespace and quotes
    items = [x.strip().strip("'").strip('"') for x in array_str.split(',') if x.strip()]
    return items

def split_dois(main_csv, doi_arrays_csv, output_common_csv, output_only_in_arrays_csv):
    # Load main CSV
    main_df = pd.read_csv(main_csv)
    main_dois = set(main_df['DOI'])
    print(f"DOIs in main CSV: {len(main_dois)}")

    # Load DOI arrays CSV
    arrays_df = pd.read_csv(doi_arrays_csv, header=None, names=['DOI_array'])

    # Flatten DOI arrays into a single list
    all_array_dois = []
    for idx, row in arrays_df.iterrows():
        doi_list = parse_doi_array(row['DOI_array'])
        all_array_dois.extend(doi_list)

    # Remove duplicates by converting to set
    all_array_dois = set(all_array_dois)
    print(f"Total unique DOIs in arrays CSV: {len(all_array_dois)}")

    # DOIs present in both
    common_dois = main_dois.intersection(all_array_dois)
    common_df = main_df[main_df['DOI'].isin(common_dois)]
    common_df.to_csv(output_common_csv, index=False)
    print(f"Saved common DOIs to {output_common_csv} ({len(common_df)} rows)")

    # DOIs only in arrays CSV (flattened, single DOI per row)
    only_in_arrays = all_array_dois - main_dois
    only_in_arrays_df = pd.DataFrame({'DOI': sorted(only_in_arrays)})
    only_in_arrays_df.to_csv(output_only_in_arrays_csv, index=False)
    print(f"Saved DOIs only in arrays CSV to {output_only_in_arrays_csv} ({len(only_in_arrays_df)} rows)")

if __name__ == "__main__":
    main_csv = "final_merged_with_abstracts_filtered.csv"
    doi_arrays_csv = "top10_paths_to_source_arrays.csv"
    output_common_csv = "top10_wAbstracts.csv"
    output_only_in_arrays_csv = "top10_woAbstracts.csv"

    split_dois(main_csv, doi_arrays_csv, output_common_csv, output_only_in_arrays_csv)
