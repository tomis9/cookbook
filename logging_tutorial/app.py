import logging.config
import json
from import_app import func

# resetting basic config set in another file
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

with open('./config.json') as f:
    config = json.load(f)

logging.config.dictConfig(config)
logger = logging.getLogger('app')

logger.debug('debug one')

func()

logger.debug('debug two')
