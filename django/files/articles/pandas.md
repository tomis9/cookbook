---
title: "pandas"
date: 2019-01-25T13:46:12+01:00
draft: false
categories: ["scratchpad"]
tags: []
---

<center>
# This is not a proper blog post yet.

pandas (TODO)

</center>

Even though I use pandas almost every day, there are certain solutions that I constantly forget about.

grouping and going back to DataFrame:
https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-object-to-dataframe

```{python}
DataFrame({'count' : df1.groupby( [ "Name", "City"] ).size()}).reset_index()
```

TODO pandas

df.replace
