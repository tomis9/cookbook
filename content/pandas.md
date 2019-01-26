---
title: "Pandas"
date: 2019-01-25T13:46:12+01:00
draft: true
categories: []
tags: []
---

Even though I use pandas almost every day, there are certain solutions that I constantly forget about.

grouping and going back to DataFrame:
https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-object-to-dataframe

```{python}
DataFrame({'count' : df1.groupby( [ "Name", "City"] ).size()}).reset_index()
```

TODO pandas
