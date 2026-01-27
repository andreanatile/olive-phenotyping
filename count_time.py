from src.models.counter import OliveCounter
from src.models.utils import plot_results,times_analyzer

model_path = "checkpoints/best.pt"
folder_path = "/mnt/c/Datasets/OlivePG/bbox_ground_truth_new"
counter=OliveCounter(model_path=model_path)

result_summary,time_summary=counter.count_folder(
    folder_path=folder_path,
    conf=0.50,
    overlap_ratio=0.2,
    slice_size=1280,
)

avg_times,max_time=times_analyzer(time_summary)

print("Average Times Summary:")
print(avg_times)
print("Max Total Counting Time Summary:")
print(max_time) 


