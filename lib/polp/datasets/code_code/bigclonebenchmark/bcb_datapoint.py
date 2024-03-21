from dataclasses import dataclass

@dataclass
class BCBDatapoint:
    func1: str
    func2: str
    label: int

    def asdict(self):
        return {
            "func1": self.func1,
            "func2": self.func2,
            "label": self.label
        }