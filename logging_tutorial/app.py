import logging

logging.basicConfig(filename='./logs.log')

logger1 = logging.getLogger(__name__)
logger1.setLevel(logging.DEBUG)


logger2 = logging.getLogger('logger2')
logger2.setLevel(logging.DEBUG)

logger1.debug('info 1')
logger2.debug('info 2')
