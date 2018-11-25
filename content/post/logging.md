---
title: "logging"
date: 2018-11-09T23:01:35+01:00
draft: false
categories: ["python"]
tags: ["draft"]
---

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

