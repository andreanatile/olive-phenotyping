from src.models.extractor import OliveExtractor
import os

if __name__ == "__main__":
    # Settings
    MODEL_PATH = "checkpoints/best.pt"
    # Use the image found earlier
    IMG_PATH = "data/Normalized/corrected/IMG_20240807_082537.jpg" 
    OUTPUT_DIR = "extracted_olives"
    
    # Initialize Extractor
    extractor = OliveExtractor(MODEL_PATH)
    
    # Run Extraction
    count = extractor.extract(
        img_path=IMG_PATH,
        output_dir=OUTPUT_DIR,
        conf=0.25,
        overlap_ratio=0.2
    )
    
    print(f"Process complete. Total extracted: {count}")
