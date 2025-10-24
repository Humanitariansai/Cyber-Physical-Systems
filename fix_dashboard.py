import os
from pathlib import Path

# Define the files to update
dashboard_dir = Path(r"c:\Users\udish\OneDrive\Documents\Udisha\Full time\Cyber-Physical Systems\Cyber-Physical-Systems\streamlit-dashboard")
files_to_update = [
    dashboard_dir / "app.py",
    dashboard_dir / "pages" / "1_Data_Analytics.py",
    dashboard_dir / "pages" / "2_ML_Models.py",
    dashboard_dir / "pages" / "3_System_Health.py"
]

for file_path in files_to_update:
    if file_path.exists():
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace use_container_width=True with width="stretch"
        updated_content = content.replace('use_container_width=True', 'width="stretch"')
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"Updated: {file_path.name}")
    else:
        print(f"File not found: {file_path}")

print("\nAll files updated successfully!")
