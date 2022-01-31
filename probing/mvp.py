class Probing:
    def __init__(self):
        self.encoders = []
        self.classifier = None
        self.probe_tasks = []


class Encoder:
    def __init__(self, model_name):
        self.encoder = None
        self.model_name = model_name


class Classifier:
    def __init__(self, cls_name):
        self.classifier = None
        self.cls_name = cls_name
