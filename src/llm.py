import sys


class LLM:
  # Types of data
  TRAINING = 'train'
  EVALUATION = 'eval'
  SPLITS = [ TRAINING, EVALUATION ]


class LLMSettings:
    OPTIONS = [ "default", "hyper" ]
    DEFAULT = OPTIONS[0]


    def __init__(self, name=DEFAULT):
        self.setParameters(name)


    def setDefault(self):
        # Model
        self.n_layer = 6
        self.n_head = 4
        self.n_embed = 64           # number of embedding dimensions
        self.n_dropout = 0.2        # i.e. 20% of nodes are removed each pass
        self.learning_rate = 1e-3   # 1e-3 for small models, 1e-4 for larger

        # Data
        self.block_size = 4         # what is the maximum context length for predictions?
        self.batch_size = 64        # how many independent sequences will we process in parallel?

    
    def setHyper(self):
        # Very deep and expensive computations here (all from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)
        self.n_layer = 6
        self.n_head = 6
        self.n_embed = 384
        self.n_dropout = 0.2
        self.learning_rate = 3e-4

        self.block_size = 64
        self.batch_size = 256


    # To be extended
    def setParameters(self, name):
        fn = f"set{name.capitalize()}"
        getattr(self, fn)()



class LLMTraining:
    OPTIONS = [ "default", "quick", "long", "forever" ]
    DEFAULT = OPTIONS[0]


    def __init__(self, name=DEFAULT):
        self.setParameters(name)


    def setDefault(self):
        self.samples_to_calculate_loss = 50
        self.steps_total = 1000
        self.steps_logging = 10
        self.report_timestep = None


    def setQuick(self):
        self.samples_to_calculate_loss = 10
        self.steps_total = 10
        self.steps_logging = 10
        self.report_timestep = None


    def setLong(self):
        self.samples_to_calculate_loss = 30
        self.steps_total = 10000
        self.steps_logging = 100
        self.report_timestep = None


    def setForever(self):
        self.samples_to_calculate_loss = 50
        self.steps_total = sys.maxsize
        self.steps_logging = 100
        self.report_timestep = 30


    def setParameters(self, name):
        self.name = name
        fn = f"set{name.capitalize()}"
        getattr(self, fn)()


    def __str__(self):
        return f"{self.name} : loss samples {self.samples_to_calculate_loss}, steps_total={self.steps_total} steps_logging={self.steps_logging}"
