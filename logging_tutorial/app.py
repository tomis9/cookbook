import logging
from import_app import func


logging.basicConfig(filename='./logs.log', level=logging.INFO)
logger = logging.getLogger('app.py')

logger.info('debug one')

func()

logger.info('debug two')
