import pandas as pd
from tqdm import tqdm
import os
import openai

# Make sure your OPENAI_API_KEY is set in your environment
# export OPENAI_API_KEY="YOUR_KEY_HERE"   (Linux/macOS)
# setx OPENAI_API_KEY "YOUR_KEY_HERE"     (Windows)

NODES_FILE = "top10_wAbstracts.csv"  # CSV with DOI and Abstract columns
PATHS_FILE = "top10_paths_to_source_arrays.csv"  # CSV with arrays of DOIs
OUTPUT_FILE = "paths_linking_sentences.csv"
ABSTRACT_COLUMN = "Abstract"  # column name for abstracts

# Initialize client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load files
nodes_df = pd.read_csv(NODES_FILE)
paths_df = pd.read_csv(PATHS_FILE)

linking_sentences = []

for i, row in tqdm(paths_df.iterrows(), total=len(paths_df)):
    try:
        # Extract DOI array from the first column
        path_str = str(row[0])
        path_dois = [doi.strip() for doi in path_str.strip("[]").split(",")]
        
        # Map DOIs to abstracts
        abstracts = []
        for doi in path_dois:
            doi = doi.replace("'", "").replace('"', '').strip()
            abstract_row = nodes_df[nodes_df['DOI'] == doi]
            if not abstract_row.empty:
                abstracts.append(abstract_row.iloc[0][ABSTRACT_COLUMN])
        
        if not abstracts:
            linking_sentences.append("No abstracts found for this path.")
            continue
        
        combined_text = " ".join(abstracts)
        
        # Call Gemini
        response = client.chat.completions.create(
            model="gemini-1.5",  # Replace with the exact Gemini model you have access to
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize in one sentence what links these papers together: {combined_text}"
                }
            ]
        )
        sentence = response.choices[0].message.content
        linking_sentences.append(sentence)
    
    except Exception as e:
        print(f"Error calling Gemini for path {i}: {e}")
        linking_sentences.append("Error generating linking sentence.")

# Save results
paths_df['Linking_Sentence'] = linking_sentences
paths_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved linking sentences to {OUTPUT_FILE}")
