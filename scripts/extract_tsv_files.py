# import os
# import json
# import pandas as pd

# # Example input variables
# dataset = "figures"
# aligned_files = f"/home/ibilgin/Dropbox/alignment_dev/text_alignment/{dataset}"
# tsv_output_directory = f"{aligned_files}/tsv_files"

# # Ensure the output directory exists
# os.makedirs(tsv_output_directory, exist_ok=True)

# # List all files in the directory
# json_files = [f for f in os.listdir(aligned_files) if f.endswith('.json')]

# print(tsv_output_directory)

# for item in json_files:
#     print(item)
#     transcript_file = os.path.join(aligned_files, item) 
#     filename, _ = os.path.splitext(item)
#     out_csv = os.path.join(tsv_output_directory, filename + ".tsv")

#     with open(transcript_file, 'r') as json_file:
#         data = json.load(json_file)

#     # Create an empty DataFrame with the necessary columns
#     df = pd.DataFrame(columns=['word', 'onset', 'offset', 'duration'])

#     if 'results' in data and 'channels' in data['results'] and data['results']['channels']:
#         alternatives = data['results']['channels'][0].get('alternatives', [])
#         if alternatives:
#             for row in alternatives[0].get('words', []):
#                 if 'word' in row:
#                     onset = row['onset']
#                     offset = row['offset']
#                     duration = offset - onset
#                     new_row = pd.DataFrame({'word': [row['word']], 'onset': [onset], 'offset': [offset], 'duration': [duration]})
#                     df = pd.concat([df, new_row], ignore_index=True)

#     # Write the DataFrame to a TSV file
#     df.to_csv(out_csv, sep='\t', index=False)
#     print(f"Created {out_csv}")


import os
import json
import pandas as pd

# Example input variables
dataset = "bourne"
aligned_files = f"/home/ibilgin/Dropbox/alignment_dev/text_alignment/{dataset}"
tsv_output_directory = f"{aligned_files}/tsv_files"

# Ensure the output directory exists
os.makedirs(tsv_output_directory, exist_ok=True)

# List all files in the directory
json_files = [f for f in os.listdir(aligned_files) if f.endswith('.json')]

print(tsv_output_directory)

for item in json_files:
    print(item)
    transcript_file = os.path.join(aligned_files, item) 
    filename, _ = os.path.splitext(item)
    out_csv = os.path.join(tsv_output_directory, filename + ".tsv")

    with open(transcript_file, 'r') as json_file:
        data = json.load(json_file)

    # Create an empty DataFrame with the necessary columns
    df = pd.DataFrame(columns=['word', 'onset', 'offset', 'duration'])

    if 'results' in data and 'channels' in data['results'] and data['results']['channels']:
        alternatives = data['results']['channels'][0].get('alternatives', [])
        if alternatives:
            for row in alternatives[0].get('words', []):
                if 'word' in row:
                    onset = row['onset']
                    offset = row['offset']
                    duration = offset - onset
                    new_row = pd.DataFrame({'word': [row['word']], 'onset': [onset], 'offset': [offset], 'duration': [duration]})

                    # Check for empty new_row before concatenation
                    if not new_row.empty:
                        df = pd.concat([df, new_row], ignore_index=True)

    # Write the DataFrame to a TSV file
    df.to_csv(out_csv, sep='\t', index=False)
    print(f"Created {out_csv}")
