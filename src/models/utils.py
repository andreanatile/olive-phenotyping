from pathlib import Path
import yaml

def parse_cfg(yaml_file_path):
    """
    Parses a YAML file and extracts the absolute paths for specified keys.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)

        base_path = Path(config.get('path'))
        if not base_path:
            raise ValueError("The 'path' key is missing or empty in the YAML file.")

        keys_to_process = ['bbox_gt', 'seg_gt']

        for key in keys_to_process:
            relative_path = config.get(key)
            if relative_path:
                # Join base and relative paths, then get the absolute path
                gt_path = base_path / relative_path
                config[key] = gt_path.resolve()
            else:
                config[key] = None

        return config

    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"An error occurred: {e}")
        return None
