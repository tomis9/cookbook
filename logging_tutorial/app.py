import logging
from import_app import func

# resetting basic config set in another file
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)
hdl = logging.FileHandler('./logs.log')
format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
hdl.setFormatter(logging.Formatter(format_str))
logger.addHandler(hdl)

logger.debug('debug one')

func()

logger.debug('debug two')
