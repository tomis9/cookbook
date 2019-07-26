---
title: "Passing arguments to scripts"
date: 2019-02-05T09:34:23+01:00
draft: false
categories: ["scratchpad"]
tags: []
---

<center>
# This is not a proper blog post yet, just my notes.

passing arguments (TODO)
</center>

[caffee](http://caffe.berkeleyvision.org/tutorial/)

## R

[A good article on passing arguments to R scripts](https://www.r-bloggers.com/passing-arguments-to-an-r-script-from-command-lines/)

You can read the docs of `commandArgs` for more info, but the general use is very simple:

```{r}
args <- commandArgs(trailingOnly = TRUE)
print(args)  # args is a vector of values
```
```{bash}
Rscript file.R one two 3
```

If `trailingOnly` is set to FALSE, args will contain some other argument values, e.g. "--slave", "--no-restore", which are usually not particularly useful and are definitely not provided by the user. `trailingOnly` set to TRUE will choose only those arugemnts that are provided byt the user.

You can also use `optparse` package, which lets you provide arguments in a more linux-like way.

```{r}
library("optparse")
 
option_list = list(
  make_option(c("-o", "--one"), type = "character", default = "first_argument"),
  make_option(c("-t", "--two"), type = "numeric", default = 2)
)
 
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

print(opt)
```

```{bash}
Rscript file.R --one first_one -t 10
```

## Python

The easiest way:

```{python}
import sys

print(str(sys.argv))
```

```
python3 files.py hi wats up
```
returns `['files.py', 'hi', 'wats', 'up']`.

and a slightly more complicated way:

```{python}
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--one", dest="one")
parser.add_argument("-o", dest="one")
parser.add_argument("--two", dest="two")
args = vars(parser.parse_args())

print(str(args))
```
