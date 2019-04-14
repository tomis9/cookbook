import logging.config
import json


with open('./config.json') as f:
    config = json.load(f)

logging.config.dictConfig(config)
logger = logging.getLogger('app')

# hdl = CustomHandler()
# logger.addHandler(hdl)

logger.debug('first')

try:
    x = 1 / 0
except ZeroDivisionError as e:
    logger.error(e)
    raise


logger.debug('second')
