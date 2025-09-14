import os
import pandas as pd
from contextus import Contextus

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
BASE_URL = os.getenv("BASE", "https://www.contextus.space")
COLLECTION_ID = os.getenv("PROJECT_ID")  # Your collection/PROJECT_ID

if not COLLECTION_ID:
    raise ValueError("PROJECT_ID must be set in your environment.")

# Initialize Contextus client
ctx = Contextus(base_url=BASE_URL)

# -----------------------------
# 2️⃣ Read your CSV
# -----------------------------
csv_file = "papers.csv"  # Replace with your CSV path
df = pd.read_csv(csv_file)

# -----------------------------
# 3️⃣ Create projects (nodes)
# -----------------------------
projects = {}
for _, row in df.iterrows():
    doi = row["DOI"]
    title = row.get("Title", "")
    abstract = row.get("Abstract", "")
    source = row.get("Source", "")
    year = row.get("Year", "")

    project = ctx.create_project(
        collection=COLLECTION_ID,
        name=doi,
        metadata={
            "title": title,
            "source": source,
            "year": year
        },
        artifacts={"abstract": abstract}
    )
    projects[doi] = project

print(f"✅ Created {len(projects)} projects (papers) in Contextus.")

# -----------------------------
# 4️⃣ Generate idea links using Gemini and create edges
# -----------------------------
# Pseudo-function for Gemini API call
def generate_idea_link(abstract_A, abstract_B):
    """
    Replace this with your actual Gemini API call.
    Example: call LLM with prompt comparing abstract_A to abstract_B.
    """
    prompt = f"Summarize in one sentence how Paper A: '{abstract_A}' relates to Paper B: '{abstract_B}' in terms of ideas."
    
    # Placeholder: replace with real Gemini call
    # idea_link = gemini.generate(prompt)
    idea_link = f"Idea link: how A relates to B based on abstracts."  # temporary placeholder
    return idea_link

# Loop through each paper and its references to generate edges
for _, row in df.iterrows():
    source_doi = row["DOI"]
    references = row.get("References", "")

    if pd.isna(references) or references.strip() == "":
        continue

    references_list = [r.strip() for r in references.split(";")]

    for target_doi in references_list:
        if target_doi in projects:
            abstract_A = row["Abstract"]
            abstract_B = df[df["DOI"] == target_doi]["Abstract"].values[0]
            
            idea_link = generate_idea_link(abstract_A, abstract_B)

            ctx.create_relation(
                source_project=projects[source_doi],
                target_project=projects[target_doi],
                relation_type="idea_link",
                metadata={"supporting_sentence": idea_link}
            )

print("✅ All idea-based edges successfully created in Contextus!")
