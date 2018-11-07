import logging


# purposely harmful logging declaration <begin
# logging.basicConfig(level=logging.INFO)
# end>

logger = logging.getLogger('imp')


def func():
    logger.debug('debug form import app')
    logger.warning('warning form import app')

