class TesterRegistry:
    def __init__(self):
        self.logger_type_classes = {}
        self.command_classes = {}

    def register(self, name: str, logger_type_class, command_class):
        self.logger_type_classes[name] = logger_type_class
        self.command_classes[name] = command_class

    def get_names(self):
        return [name for name in self.logger_type_classes.keys()]


# make global tester registry
tester_registry = TesterRegistry()
