---
title: "logging"
date: 2018-10-20T00:15:21+02:00
draft: false
image: "python.png"
categories: ["python", "R"]
tags: ["python", "R", "logging"]
---

## 1. What is logging and why would you use it?

* Logging, in general, provides information about the execution of a program outside, e.g. to stdout or to a file. Why would that be useful?

* You may get the information of how and when the program was executed, e.g. who was using it's functionalities and if all the parts of your program finished correctly.

`logging` module, which is available in python's standard library, contains various functions and objects, which make logging easy and standardised.

## Python

##3 2. "Hello World" examples

#### Basic configuration

```{python}
import logging

format = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(filename='logs.log', level=logging.INFO, 
        logging.format=format)

logging.debug('some debug')
logging.info('some info')
logging.warning('another warning')
logging.error('some error')
```

or using a `logger`:

```{python}
import logging

format = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(filename='logs.log', level=logging.INFO, 
        logging.format=format)
logger = logging.getLogger(__name__)

logging.debug('some debug')
logging.info('some info')
logging.warning('another warning')
logging.error('some error')
```

#### Not so basic configuration

```{python}
import logging

logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)

hdl = logging.FileHandler('./logs.log')
format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
hdl.setFormatter(logging.Formatter(format_str))
logger.addHandler(hdl)

logger.debug('some debug')
```

#### Keeping configuration in a dictionary

```{python}
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
logger = logging.getLogger(__name__)

logger.debug('some debug')
```

#### Keeping configuration in a separate file

*app.py*

```{python, eval = FALSE, python.reticulate = FALSE}
import logging.config
import json
from import_app import func

with open('./config.json') as f:
    config = json.load(f)

logging.config.dictConfig(config)
logger = logging.getLogger('app')

logger.debug('some debug')
```

*config.json*
```{json}
{
  "version": 1,
  "disable_existing_loggers": "False",
  "formatters": {
    "standard": {
      "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    }
  },
  "handlers": {
    "app": {
      "formatter": "standard",
      "class": "logging.FileHandler",
      "filename": "./logs.log"
    },
    "sub": {
      "formatter": "standard",
      "class": "logging.FileHandler",
      "filename": "./logs_imp.log"
    }
  },
  "loggers": {
    "app": {
      "level": "DEBUG",
      "handlers": ["app"]
    },
    "imp": {
      "level": "INFO",
      "handlers": ["sub"]
    }
  }
}
```

#### Tips and tricks

* resetting basic config set in another file

```{python}
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
```

#### Useful links
https://www.toptal.com/python/in-depth-python-logging

## R

futile.logger

