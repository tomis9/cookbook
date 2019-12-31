import logging


class CustomHandler(logging.StreamHandler):

    def __init__(self, value):
        super().__init__()
        print(value)

    def emit(self, record):
        msg = self.format(record)
        print("elo", msg)
