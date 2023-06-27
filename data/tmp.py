with open('data/ultra_eval.txt', 'r') as file:
    lines = file.readlines()

# Initialize an empty list to store our JSON objects
json_list = []

# Loop through each line of the file
for line in lines:
    # Check if the line starts and ends with "--"
    if line.startswith("--") and line.endswith("--\n"):
        # Extract the text between the "--"
        class_name = line.strip("--\n")
    else:
        # Use the entire line as the value for the "data" key
        data = line.strip()
        # Create a dictionary object with the "class" and "data" keys
        json_obj = {"class": class_name, "data": data}
        # Append the dictionary object to our list
        json_list.append(json_obj)

# Convert the list of dictionary objects to a JSON string
json_str = json.dumps(json_list)

# Print the JSON string
print(json_str)