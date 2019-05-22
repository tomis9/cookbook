---
title: "logging"
date: 2018-10-20T00:15:21+02:00
draft: false
image: "logging.jpg"
categories: ["python", "R", "data-engineering"]
tags: ["python", "R", "logging", "data-engineering"]
---

<center>

**Contents:**

[1. What is logging and why would you use it?](#what) 

[2. "Hello World" examples](#hello) 

[Python](#python)

[R](#r)

[3. Subjects still to cover](#subjects)

</center>



## 1. What is logging and why would you use it? {#what}

* Logging, in general, sends information about the execution of a program to the outside of the program, e.g. to stdout or to a file. Why would that be useful?

* You may get the information of how and when the program was executed, e.g. who was using it's functionalities and if all the pieces of your program finished correctly.

https://logmatic.io/blog/beyond-application-monitoring-discover-logging-best-practices/

## 2. "Hello World" examples {#hello}

## Python {#python}

`logging` module, which is available in python's standard library, contains various functions and objects, which make logging easy and standardised.

#### Basic configuration

```{python}
import logging

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='logs.log', level=logging.INFO, format=format)

logging.debug('some debug')
logging.info('some info')
logging.warning('another warning')
logging.error('some error')
```

or using a `logger`:

```{python}
import logging

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
hdl_file = logging.FileHandler('./logs.log')
hdl_file.setFormatter(logging.Formatter(format))
hdl_stdout = logging.StreamHandler(sys.stdout)
hdl_stdout.setFormatter(logging.Formatter(format))

logger.addHandler(hdl_file)
logger.addHandler(hdl_stdout)

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

```{python}
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
```
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

## R {#r}

There are several packages for logging in R, e.g:

* `futile.logger`,

* `logging`,

* `r-logging`.

I will discuss only `futile.logger`, because this is tne only one I've been using by now. However I the nearest future I am planning to give `logging` a try, as it seems to be analogical to python's standard `logging`.

#### A few "Hello World" examples

This is how you print basic log messages:

```{r}
library(futile.logger)
flog.info("My first log statement with futile.logger")
flog.warn("This statement has higher severity")
flog.fatal("This one is really scary")
```

You can easily create log messages dynamically:
```{r}
flog.info("This is my %s log statement, %s", 'second', 'ever')
```

As in python's `logging`, you can set the level (in `futile.logger` it's called *threshold*) of the messages:
```{r}
flog.threshold(WARN)
flog.info("Log statement %s is hidden!", 3)
flog.warn("However warning messages will still appear")
```

Instead of printing log messages to the standard output, you can append them to a file. You can also name the handler for future reference, e.g. "data.io" as in the following exaple:
```{r}
flog.appender(appender.file("data.io.log"), name="data.io")
```

There are two types of appenders: standard output (known as *console*) and a file.
```{r}
appender.console() 
appender.file()
```


## 3. Subjects still to cover {#subjects}

* ftry (TODO) how does ftry cope with exceptions?

Simple, isn't it?
