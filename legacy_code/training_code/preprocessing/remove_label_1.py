import os

def remove_object_1_lines(txt_file_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    # Filter out lines corresponding to object 1 (class index starts from 0)
    lines = [line for line in lines if float(line.split()[0]) != 1.0]

    # Write the filtered lines back to the file
    with open(txt_file_path, 'w') as file:
        file.writelines(lines)



# Directory containing YOLO boundary box annotation files
annotation_dir = 'data/original/T2-TRI/labels'

# Iterate through each .txt file in the directory
for filename in os.listdir(annotation_dir):
    if filename.endswith('.txt'):
        txt_file_path = os.path.join(annotation_dir, filename)
        remove_object_1_lines(txt_file_path)