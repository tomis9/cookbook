import logging

# purposedly harmful logging declaration <begin
logging.basicConfig(level=logging.INFO)
# end>

logger = logging.getLogger('import_app.py')
logger.setLevel(logging.DEBUG)


def func():
    logger.debug('hello form import app')
