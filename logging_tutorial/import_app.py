import logging
import sys


# purposely harmful logging declaration <begin
logging.basicConfig(level=logging.INFO)
# end>

logger = logging.getLogger('imp')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler('./logs_imp.log'))

stdoutHandler = logging.StreamHandler(sys.stdout)
stdoutHandler.setLevel(logging.WARNING)
logger.addHandler(stdoutHandler)


def func():
    logger.debug('debug form import app')
    logger.warning('warning form import app')

