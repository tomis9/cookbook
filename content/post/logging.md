---
title: "logging"
date: 2018-10-20T00:15:21+02:00
draft: false
image: "python.png"
categories: ["python"]
tags: ["draft", "python", "logging"]
---

## 1. What is logging and why would you use it?

* Logging, in general, provides information about the execution of a program outside, e.g. to stdout or to a file. Why would that be useful?

* You may get the information if all the parts of the program executed correctly, for example where and when errors occured.

* You may get the information of how and when the program was executed, e.g. who was using it's functionalities.

`logging` module, which is one of python's standard modules, provides you with a couple of functios and objects, which make logging easy and standardised.

## 2. "Hello World" examples

* ### Basic configuration

```{python}
import logging

logging.basicConfig(filename='logs.log', level=logging.INFO)

logging.warning('some warning')
logging.warning('another warning')
```

or using a `logger`:

```{python}
import logging

logging.basicConfig(filename='logs.log', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.warning('some warning')
logger.warning('another warning')
```

Why it'a a good idea to use `logger` instead of a basic `logging`?


* ### Not so basic configuration

* ### Keeping configuration in a separate file

app.py
```{python, eval = FALSE, python.reticulate = FALSE}
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

```

import_app.py
```{python, eval = FALSE, python.reticulate = FALSE}
import logging


# purposely harmful logging declaration <begin
# logging.basicConfig(level=logging.INFO)
# end>

logger = logging.getLogger('imp')


def func():
    logger.debug('debug form import app')
    logger.warning('warning form import app')

```

config.json
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
https://www.toptal.com/python/in-depth-python-logging
