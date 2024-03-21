from collections import defaultdict
import json

class MetricLogger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricLogger, cls).__new__(cls)
            cls._instance.log = defaultdict()
        return cls._instance


    def update(self, key, new_val):
        """
        Used for keys with non-list values
        """
        self.log[key] = new_val
    
    def list_add(self, key, value):
        """
        Used for keys that contain list values
        """
        self.log[key].append(value)

    def create(self, key, value):
        """
        Creates new key in the logger
        """
        self.log[key] = value

    def get(self, key):
        return self.log[key]

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.log, f, indent=4)