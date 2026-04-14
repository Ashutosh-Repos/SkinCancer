import os
import sys
import glob
import shutil
import zipfile
import argparse
import subprocess

def download_dataset(output_dir='data'):
    """
    Download HAM10000 dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
    """
    try:
        import kaggle
    except ImportError:
        print("Error: kaggle package not installed.")
        print("Install with: pip install kaggle")
        sys.exit(1)
    
    # Check for credentials: either KAGGLE_API_TOKEN env var or kaggle.json file
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    has_token = os.environ.get('KAGGLE_API_TOKEN') is not None
    
    if not has_token and not os.path.exists(kaggle_json):
        print("Error: Kaggle API credentials not found.")
        print("\nOption 1 — API Token (recommended):")
        print("  Set environment variable: export KAGGLE_API_TOKEN=KGAT_xxx")
        print("\nOption 2 — Legacy kaggle.json:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Create New API Token → downloads kaggle.json")
        print(f"  3. Move it to {kaggle_dir}/")
        sys.exit(1)
    
    print("Downloading HAM10000 dataset from Kaggle...")
    print("This may take several minutes (2.6 GB download)...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset (with error checking)
    result = subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', 'kmader/skin-cancer-mnist-ham10000', '-p', output_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error downloading dataset: {result.stderr}")
        sys.exit(1)
    
    print("\nExtracting files...")
    
    # Extract main zip
    main_zip = os.path.join(output_dir, 'skin-cancer-mnist-ham10000.zip')
    if os.path.exists(main_zip):
        with zipfile.ZipFile(main_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(main_zip)
        print("  Extracted main archive")
    
    # Create target images directory
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Strategy 1: Extract sub-zip files (older Kaggle format)
    for part in [1, 2]:
        part_zip = os.path.join(output_dir, f'HAM10000_images_part_{part}.zip')
        if os.path.exists(part_zip):
            with zipfile.ZipFile(part_zip, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            os.remove(part_zip)
            print(f"  Extracted image part {part} (from zip)")
    
    # Strategy 2: Move images from directories (newer Kaggle format)
    for part in [1, 2]:
        part_dir = os.path.join(output_dir, f'HAM10000_images_part_{part}')
        if os.path.isdir(part_dir):
            moved = 0
            for f in os.listdir(part_dir):
                if f.endswith('.jpg'):
                    shutil.move(os.path.join(part_dir, f), os.path.join(images_dir, f))
                    moved += 1
            shutil.rmtree(part_dir, ignore_errors=True)
            print(f"  Moved {moved} images from part {part} directory")
    
    # Strategy 3: Find any loose .jpg files anywhere under data/ and move to images/
    for jpg_file in glob.glob(os.path.join(output_dir, '**', '*.jpg'), recursive=True):
        if os.path.dirname(jpg_file) != images_dir:
            shutil.move(jpg_file, os.path.join(images_dir, os.path.basename(jpg_file)))
    
    # Remove CSV files we don't need
    csv_files = [
        'hmnist_8_8_RGB.csv',
        'hmnist_8_8_L.csv',
        'hmnist_28_28_RGB.csv',
        'hmnist_28_28_L.csv'
    ]
    
    for csv_file in csv_files:
        csv_path = os.path.join(output_dir, csv_file)
        if os.path.exists(csv_path):
            os.remove(csv_path)
    
    # Verify
    jpg_count = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    print(f"\nDataset ready! Found {jpg_count} images in {os.path.abspath(images_dir)}")
    if jpg_count < 10000:
        print(f"WARNING: Expected ~10015 images but found {jpg_count}.")
        print("Check the data/ directory structure manually.")

def main():
    parser = argparse.ArgumentParser(
        description='Download HAM10000 dataset from Kaggle'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory for dataset'
    )
    
    args = parser.parse_args()
    download_dataset(args.output)

if __name__ == '__main__':
    main()

