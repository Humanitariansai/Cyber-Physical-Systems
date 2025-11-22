import os
import re
from pathlib import Path

def remove_emojis(text):
    """Remove all emoji characters from text"""
    # Comprehensive emoji pattern
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002700-\U000027BF"  # Dingbats
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def process_file(file_path):
    """Process a single file to remove emojis"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = remove_emojis(content)
        
        if content != new_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    project_root = Path(__file__).parent
    extensions = ['.py', '.md', '.sh', '.bat']
    exclude_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv'}
    
    files_cleaned = 0
    files_processed = 0
    
    for ext in extensions:
        for file_path in project_root.rglob(f'*{ext}'):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue
            
            files_processed += 1
            if process_file(file_path):
                files_cleaned += 1
                print(f"Cleaned: {file_path.name}")
    
    print(f"\nTotal files processed: {files_processed}")
    print(f"Total files cleaned: {files_cleaned}")

if __name__ == "__main__":
    main()
