import logging
from import_app import func

# resetting basic config set in another file
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler('./logs.log'))

logger.debug('debug one')

func()

logger.debug('debug two')
