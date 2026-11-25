from typing import List, Dict, Any

class BreastCancerData:
    def __init__(self, tumor_size: float, age: int, other_features: Dict[str, Any]):
        self.tumor_size = tumor_size
        self.age = age
        self.other_features = other_features

class ModelOutput:
    def __init__(self, prediction: float, confidence: float):
        self.prediction = prediction
        self.confidence = confidence

def prepare_data(data: List[BreastCancerData]) -> List[Dict[str, Any]]:
    processed_data = []
    for entry in data:
        processed_entry = {
            "tumor_size": entry.tumor_size,
            "age": entry.age,
            **entry.other_features
        }
        processed_data.append(processed_entry)
    return processed_data