import logging
from import_app import func


logging.basicConfig(filename='./logs.log', level=logging.INFO)

logging.info('debug one')

func()
