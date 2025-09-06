import os
import json


def preprocess_json(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    # Find the first '{' and the last '}' to keep them
    first_brace_index = data.find('{')
    last_brace_index = data.rfind('}')

    if first_brace_index == -1 or last_brace_index == -1 or first_brace_index == last_brace_index:
        print("Invalid JSON structure.")
        return

    # Extract the beginning, middle, and end
    start = data[:first_brace_index + 1]
    end = data[last_brace_index:]
    middle = data[first_brace_index + 1:last_brace_index]

    # Remove all remaining '{' and '}'
    cleaned_middle = middle.replace('{', '').replace('}', '')

    # Combine and write back
    cleaned_data = start + cleaned_middle + end
    return cleaned_data


root_path = ""
input_files = [
    os.path.join(root_path, f"result_rank0.json"),
    os.path.join(root_path, f"result_rank1.json"),
    os.path.join(root_path, f"result_rank2.json"),
]

combined_data = {}

for file in input_files:
    preprocessed_content = preprocess_json(file)
    data = json.loads(preprocessed_content)  # Now the content should be valid JSON
    for video_name, video_data in data.items():
        if video_name not in combined_data:
            combined_data[video_name] = []
        combined_data[video_name].extend(video_data)

sorted_combined_data = {k: combined_data[k] for k in sorted(combined_data)}

with open(os.path.join(root_path, f"result.json"), 'w') as f:
    json.dump(sorted_combined_data, f, indent=4)
