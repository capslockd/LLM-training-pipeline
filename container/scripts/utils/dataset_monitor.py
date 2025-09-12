import os

import yaml


class DatasetCheckpoint:
    def __init__(self, checkpoint_file="config/dataset_checkpoint.yml"):
        self.checkpoint_file = checkpoint_file
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                return yaml.safe_load(f) or {"datasets": {}}
        return {"datasets": {}}

    def save(self):
        with open(self.checkpoint_file, "w") as f:
            yaml.safe_dump(self.state, f)

    def get_last_index(self, dataset_key):
        return self.state.get("datasets", {}).get(dataset_key, {}).get("last_index", 0)

    def update(self, dataset_key, last_index, output_file):
        if "datasets" not in self.state:
            self.state["datasets"] = {}

        self.state["datasets"][dataset_key] = {
            "last_index": last_index,
            "output_file": output_file,
        }
        self.save()
