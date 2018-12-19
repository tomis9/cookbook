---
title: "redis"
date: 2018-10-21T20:49:20+02:00
draft: false
crategories: ["scratchpad"]
---

<center>
# This is not a proper blog post yet, just my notes.

redis (TODO)
</center>

```{python}
import redis
import pandas as pd

# https://www.youtube.com/watch?v=Hbt56gFj998
# https://redis-py.readthedocs.io/en/latest/
# open up a redis-server session in redis/src/redis-server

redis_host = "localhost"
redis_port = 6379
redis_password = ""

r = redis.StrictRedis(host=redis_host, port=redis_port,
                      password=redis_password, decode_responses=True)

r.flushall()

# save data to redis
d = {key: str(value) for key, value in zip(list('abcdefghij'), range(10))}
for key, value in d.items():
    r.set(key, value)

# get data from redis
vals = dict()
for key in r.keys():
    vals[key] = r.get(key)


assert vals == d

result = pd.DataFrame(list(vals.items()), columns=['key', 'value'])
```
