import logging

logging.basicConfig(filename='logs.log', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.warning('some warning')

logger.warning('another warning')

1 / 0

logger.warning('theoretically after 1/0')
