import os
import shutil
import json


def empty_subfolders(root_folder):
    """Empty all subfolders while keeping the folder structure intact."""

    if not os.path.exists(root_folder):
        print(f"Root folder does not exist: {root_folder}")
        return

    deleted_count = 0
    folders_processed = 0

    # Walk through all subdirectories
    for root, dirs, files in os.walk(root_folder):
        # Skip the root folder itself, only process subfolders
        if root == root_folder:
            continue

        folders_processed += 1
        subfolder_name = os.path.basename(root)

        # Delete all files in this subfolder
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"Deleted: {file} from {subfolder_name}/")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        # Delete any nested subfolders (if any)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                shutil.rmtree(dir_path)
                print(f"Deleted subfolder: {dir_name} from {subfolder_name}/")
            except Exception as e:
                print(f"Error deleting directory {dir_path}: {e}")

    print(f"\nSummary:")
    print(f"- Processed {folders_processed} subfolders")
    print(f"- Deleted {deleted_count} files")
    print(f"- All subfolders kept intact and empty")


# Load config with fallback
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    input_folder = os.path.expanduser(config['input_folder'])
except FileNotFoundError:
    # Default fallback when config.json doesn't exist
    input_folder = "input_images"

print(f"Emptying all subfolders in: {input_folder}")
print("This will delete ALL files in ALL subfolders!")
confirmation = input("Are you sure you want to continue? (yes/no): ")

if confirmation.lower() in ['yes', 'y']:
    empty_subfolders(input_folder)
    print("Done!")
else:
    print("Operation cancelled.")
