import datetime
import shutil
import yaml
import pandas as pd 
from pathlib import Path
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

class KFoldSplitter: 
    def __init__(self, n_splits=5, random_state=20):
        self.n_splits = n_splits 
        self.random_state = random_state
    
    def generate_feature_vectors(self, dataset_path: str, yaml_path: str):
        """Generates feature vectors for object detection dataset."""
        dataset_path = Path(dataset_path)
        labels = sorted(dataset_path.rglob("labels/*.txt"))  # all data in 'labels'
        
        with open(yaml_path, encoding="utf8") as y:
            classes = yaml.safe_load(y)["names"]
        cls_idx = sorted(classes.keys())

        index = [label.stem for label in labels]
        labels_df = pd.DataFrame([], columns=cls_idx, index=index)

        for label in labels:
            lbl_counter = Counter()
            with open(label) as lf:
                lines = lf.readlines()

            for line in lines:
                # classes for YOLO label uses integer at first position of each line
                lbl_counter[int(line.split(" ", 1)[0])] += 1

            labels_df.loc[label.stem] = lbl_counter

        labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`
        return labels_df, labels, classes

    def create_kfold_splits(self, dataset_path: str, yaml_path: str, save_dir: str = None, stratified: bool = False):
        """Generates K-Fold splits and copies images and labels."""
        dataset_path = Path(dataset_path)
        labels_df, labels, classes = self.generate_feature_vectors(dataset_path, yaml_path)
        
        if stratified:
            # We use the most frequent class in each image as a proxy for stratification
            y = labels_df.idxmax(axis=1)
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            kfolds = list(kf.split(labels_df, y))
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            kfolds = list(kf.split(labels_df))

        folds = [f"split_{n}" for n in range(1, self.n_splits + 1)]
        index = labels_df.index
        folds_df = pd.DataFrame(index=index, columns=folds)

        for i, (train, val) in enumerate(kfolds, start=1):
            folds_df.loc[labels_df.iloc[train].index, f"split_{i}"] = "train"
            folds_df.loc[labels_df.iloc[val].index, f"split_{i}"] = "val"

        supported_extensions = [".jpg", ".jpeg", ".png"]
        images = []
        for ext in supported_extensions:
            images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

        if save_dir is None:
            save_path = dataset_path.parent / f"{datetime.date.today().isoformat()}_{self.n_splits}-Fold_Cross-val"
        else:
            save_path = Path(save_dir)
            
        save_path.mkdir(parents=True, exist_ok=True)
        ds_yamls = []

        for split in folds_df.columns:
            split_dir = save_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
            (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

            dataset_yaml = split_dir / f"{split}_dataset.yaml"
            ds_yamls.append(dataset_yaml)

            with open(dataset_yaml, "w") as ds_y:
                yaml.safe_dump(
                    {
                        "path": split_dir.absolute().as_posix(),
                        "train": "train",
                        "val": "val",
                        "names": classes,
                    },
                    ds_y,
                )

        # Map labels by stem for robustness in case images and labels lengths differ slightly
        label_dict = {label.stem: label for label in labels}
        
        for image in tqdm(images, total=len(images), desc="Copying files"):
            if image.stem not in folds_df.index:
                continue
            
            label = label_dict.get(image.stem)
            if not label:
                continue

            for split, k_split in folds_df.loc[image.stem].items():
                img_to_path = save_path / split / k_split / "images"
                lbl_to_path = save_path / split / k_split / "labels"

                shutil.copy(image, img_to_path / image.name)
                shutil.copy(label, lbl_to_path / label.name)

        folds_df.to_csv(save_path / "kfold_datasplit.csv")
        return ds_yamls

if __name__=="__main__":
    splitter=KFoldSplitter()
    # Fixed the missing leading '/' in the dataset_path
    dataset_path = "/home/girobat/Pictures/puppy/"
    yaml_path = "/home/girobat/Pictures/puppy/puppy.yaml"
    
    # generate_feature_vectors just returns the dataframes. 
    # To actually find images and create splits, we should call create_kfold_splits.
    # splitter.generate_feature_vectors(dataset_path, yaml_path)
    splitter.create_kfold_splits(dataset_path, yaml_path, "./puppy/kfold", stratified=True)