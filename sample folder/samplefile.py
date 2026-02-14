import json
import csv
from datetime import datetime
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    """Abstract base class for data processing."""
    
    @abstractmethod
    def process(self, data):
        pass


class CSVProcessor(DataProcessor):
    """Process CSV files."""
    
    def process(self, filepath: str) -> List[Dict]:
        """Read and process CSV file."""
        try:
            with open(filepath, 'r') as file:
                reader = csv.DictReader(file)
                return [row for row in reader]
        except FileNotFoundError:
            print(f"File {filepath} not found.")
            return []


class JSONProcessor(DataProcessor):
    """Process JSON files."""
    
    def process(self, filepath: str) -> Dict:
        """Read and process JSON file."""
        try:
            with open(filepath, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing JSON: {e}")
            return {}


class DataAnalyzer:
    """Analyze processed data."""
    
    def __init__(self, data: List[Dict]):
        self.data = data
        self.timestamp = datetime.now()
    
    def get_summary(self) -> Dict:
        """Generate data summary."""
        return {
            "total_records": len(self.data),
            "processed_at": self.timestamp.isoformat(),
            "keys": list(self.data[0].keys()) if self.data else []
        }
    
    def filter_by_key(self, key: str, value) -> List[Dict]:
        """Filter data by key-value pair."""
        return [row for row in self.data if row.get(key) == value]


class Pipeline:
    """Data processing pipeline."""
    
    def __init__(self):
        self.processors = {}
        self.data = None
    
    def register_processor(self, name: str, processor: DataProcessor):
        """Register a processor."""
        self.processors[name] = processor
    
    def execute(self, processor_name: str, filepath: str) -> Optional[Dict]:
        """Execute pipeline."""
        if processor_name not in self.processors:
            print(f"Processor {processor_name} not found.")
            return None
        
        self.data = self.processors[processor_name].process(filepath)
        return {"status": "success", "data_count": len(self.data) if self.data else 0}


def main():
    """Main entry point."""
    pipeline = Pipeline()
    pipeline.register_processor("csv", CSVProcessor())
    pipeline.register_processor("json", JSONProcessor())
    
    result = pipeline.execute("csv", "sample.csv")
    print(result)


if __name__ == "__main__":
    main()