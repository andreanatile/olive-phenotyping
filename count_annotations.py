import os
import argparse
from pathlib import Path

def main(args):
    folder_path = Path(args.folder_path)
    
    if not folder_path.exists():
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Find all .txt files recursively (this easily handles train/val/test subdirectories)
    txt_files = list(folder_path.rglob("*.txt"))
    
    total_instances = 0
    total_files_with_annotations = 0
    
    for txt_file in txt_files:
        # Skip common non-label files that might be in a YOLO dataset
        if txt_file.name in ["classes.txt", "readme.txt"]:
            continue
            
        try:
            with open(txt_file, 'r') as f:
                # Count lines that actually have content (ignore blank lines)
                lines = [line for line in f if line.strip()]
                num_lines = len(lines)
                
                total_instances += num_lines
                if num_lines > 0:
                    total_files_with_annotations += 1
                    
        except Exception as e:
            print(f"Could not read {txt_file}: {e}")

    print("\n" + "="*40)
    print("      ANNOTATION COUNT SUMMARY      ")
    print("="*40)
    print(f"Scanned Folder: {folder_path.resolve()}")
    print(f"Total .txt files processed: {len(txt_files)}")
    print(f"Files containing annotations: {total_files_with_annotations}")
    print("-" * 40)
    print(f"🎯 TOTAL ANNOTATED OLIVES: {total_instances:,}")
    print("="*40)
    print("\nUse this number to confidently explain how much manual work was")
    print("bypassed thanks to your indirect knowledge transfer strategy!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count total YOLO annotations (instances) in a folder of text files.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing YOLO .txt label files (can include train/val subdirectories).")
    
    args = parser.parse_args()
    main(args)
