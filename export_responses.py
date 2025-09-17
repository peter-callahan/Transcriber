import os
import shutil
import json

# Load config with fallback
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    output_dir = config.get('output_folder', './markdown_output')
except FileNotFoundError:
    # Default fallback when config.json doesn't exist
    output_dir = "./markdown_output"

# Expand tilde (~) for home directory if present
output_dir = os.path.expanduser(output_dir)

# Create the output directory
os.makedirs(output_dir, exist_ok=True)

# Load the JSON data
with open("responses.json", "r") as json_file:
    responses = json.load(json_file)

# Iterate through each response
for uuid, response_data in responses.items():

    # Save the markdown content with line breaks for better structure
    if response_data.get('is_valid_json', False):
        entry_data = response_data['metadata']
        
        # Handle missing or None values with safe defaults
        title = entry_data.get('title') or 'Untitled'
        date = entry_data.get('date') or 'Unknown_Date'
        tags = entry_data.get('tags') or []
        transcription = entry_data.get('transcription') or 'No transcription available'
        
        # Create safe folder name with proper null handling
        safe_title = title.replace(' ', '_') if title else 'Untitled'
        safe_date = date.replace(' ', '_') if date else 'Unknown_Date'
        folder_name = f"{safe_title}_{safe_date}"
        
        markdown_content = f"# {title}\n\n"
        if tags:
            markdown_content += f"**Tags:** {' '.join([f'#{tag}' for tag in tags])}\n\n"
        markdown_content += f"**Date:** {date}\n\n"

        # Add upload date if it exists
        if 'upload_date' in entry_data and entry_data['upload_date']:
            markdown_content += f"**Upload Date:** {entry_data['upload_date']}\n\n"

        markdown_content += f"{transcription}\n"
    else:
        folder_name = os.path.splitext(os.path.basename(uuid))[
            0]  # Use the UUID as fallback
        markdown_content = response_data["choices"][0]["message"]["content"]

    # Ensure unique folder name
    original_folder_name = folder_name
    counter = 2
    while os.path.exists(os.path.join(output_dir, folder_name)):
        folder_name = f"{original_folder_name}_{counter}"
        counter += 1
    
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    markdown_file_path = os.path.join(folder_path, f"{folder_name}.md")
    with open(markdown_file_path, "w") as markdown_file:
        markdown_file.write(markdown_content)

    # Create an 'images' subfolder within the UUID folder
    images_folder_path = os.path.join(folder_path, "images")
    os.makedirs(images_folder_path, exist_ok=True)

    # Get image paths from response data
    image_paths = response_data.get("image_paths", [])
    
    print(f"Debug: Found {len(image_paths)} image paths for {folder_name}")
    print(f"Debug: Image paths: {image_paths}")

    # Copy all relevant images to the 'images' subfolder
    for image_path in image_paths:
        print(f'Debug: Processing image path: {image_path}')
        
        # Handle both absolute and relative paths
        if os.path.exists(image_path):
            try:
                shutil.copy2(image_path, images_folder_path)
                print(f'Debug: Successfully copied {os.path.basename(image_path)}')
            except Exception as e:
                print(f'Debug: Failed to copy {image_path}: {e}')
        else:
            print(f'Debug: Image path does not exist: {image_path}')
            # Try without the ./ prefix if it exists
            clean_path = image_path.lstrip('./')
            if os.path.exists(clean_path):
                try:
                    shutil.copy2(clean_path, images_folder_path)
                    print(f'Debug: Successfully copied {os.path.basename(clean_path)} (cleaned path)')
                except Exception as e:
                    print(f'Debug: Failed to copy cleaned path {clean_path}: {e}')
            else:
                print(f'Debug: Neither original nor cleaned path exists: {image_path} -> {clean_path}')

print("Each JSON element has its own folder in markdown_output.")
