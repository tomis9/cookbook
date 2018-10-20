import logging
from import_app import func

# resetting basic config set in another file
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename='./logs.log', level=logging.INFO)
logger = logging.getLogger('app.py')
logger.setLevel(logging.INFO)

logger.info('debug one')

func()

logger.info('debug two')
