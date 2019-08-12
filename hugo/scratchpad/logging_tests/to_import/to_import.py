import logging

logger = logging.getLogger(__name__)


def add(a, b):
    logger.info("hello from to_import")
    print(__name__)
    return a + b
