import json
import os
import re


class PatternManager:
    def __init__(self, file_path='patterns/patterns.json'):
        self.file_path = file_path
        self.patterns = self.load_patterns()

    def load_patterns(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                return json.load(file)
        return {"HW": [], "SW": []}

    def save_patterns(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.patterns, file, indent=4)

    def add_pattern(self, label, pattern):
        if label in self.patterns:
            self.patterns[label].append(pattern)
            self.save_patterns()

    def classify_with_patterns(self, text):
        for label, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return label
        return None

    def learn_from_text(self, text, label):
        if label in self.patterns:
            self.patterns[label].append(text)
            self.save_patterns()
