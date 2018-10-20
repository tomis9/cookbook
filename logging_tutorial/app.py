import logging.config
from import_app import func

# resetting basic config set in another file
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'app': {
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': './logs.log',
        },
        'sub': {
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': './logs_imp.log',
        },
    },
    'loggers': {
        'app': {
            'level': 'DEBUG',
            'handlers': ['app'],
        },
        'imp': {
            'level': 'INFO',
            'handlers': ['sub'],
        },
    }
}
logging.config.dictConfig(config)
logger = logging.getLogger('app')

logger.debug('debug one')

func()

logger.debug('debug two')
