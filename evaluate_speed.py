import argparse
import numpy as np
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO inference speed on a folder of images.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model .pt file")
    parser.add_argument("--source", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for inference (default: 1)")
    parser.add_argument("--device", type=str, default="0", help="Device to use, e.g., '0' for GPU or 'cpu' (default: 0)")
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model}")
    model = YOLO(args.model)
    
    print(f"Running inference on: {args.source}")
    
    # YOLO predict automatically extracts timing metrics for every image
    results = model.predict(
        source=args.source, 
        imgsz=args.imgsz, 
        conf=args.conf, 
        batch=args.batch,
        device=args.device,
        verbose=False # Set to True if you want YOLO's default per-image logging
    )
    
    if not results:
        print("No images were processed. Please check the source folder.")
        return
        
    # Extract speed metrics from each Result object
    preprocess_times = [r.speed['preprocess'] for r in results]
    inference_times = [r.speed['inference'] for r in results]
    postprocess_times = [r.speed['postprocess'] for r in results]
    
    avg_prep = np.mean(preprocess_times)
    avg_inf = np.mean(inference_times)
    avg_post = np.mean(postprocess_times)
    
    total_avg = avg_prep + avg_inf + avg_post
    
    print("\n" + "="*45)
    print(f" Processed {len(results)} images in folder.")
    print("="*45)
    print(" Average Times Per Image (ms):")
    print(f"   Preprocessing:  {avg_prep:.2f} ms")
    print(f"   Inference:      {avg_inf:.2f} ms")
    print(f"   Postprocessing: {avg_post:.2f} ms")
    print("-" * 45)
    print(f"   Total per img:  {total_avg:.2f} ms")
    print(f"   Throughput:     {1000 / total_avg:.2f} FPS")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
