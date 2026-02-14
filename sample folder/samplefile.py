import json
import csv
from datetime import datetime
from typing import List, Dict
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def process(self, data): pass

class CSVProcessor(DataProcessor):
    def process(self, filepath: str) -> List[Dict]:
        try:
            with open(filepath) as f:
                return list(csv.DictReader(f))
        except FileNotFoundError:
            print(f"File {filepath} not found.")
            return []

class JSONProcessor(DataProcessor):
    def process(self, filepath: str) -> Dict:
        try:
            with open(filepath) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: {e}")
            return {}

class DataAnalyzer:
    def __init__(self, data: List[Dict]):
        self.data = data
        self.timestamp = datetime.now()
    
    def get_summary(self) -> Dict:
        return {
            "keys": list(self.data[0].keys()) if self.data else []
        }
    
    def filter_by_key(self, key: str, value) -> List[Dict]:
        return [row for row in self.data if row.get(key) == value]

class Pipeline:
    def __init__(self):
        self.processors = {}
        self.data = None
    
    def register_processor(self, name: str, processor: DataProcessor):
        self.processors[name] = processor
    
    def execute(self, processor_name: str, filepath: str) -> Dict:
        if processor_name not in self.processors:
            return {"status": "error", "message": f"Processor {processor_name} not found."}
        self.data = self.processors[processor_name].process(filepath)
        return {"status": "success", "data_count": len(self.data)}

if __name__ == "__main__":
    pipeline = Pipeline()
# Register processors
    pipeline.register_processor("csv", CSVProcessor())