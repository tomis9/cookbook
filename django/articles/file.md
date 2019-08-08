---
title: "Rmarkdown total basics"
date: 2018-11-09T23:01:35+01:00
draft: false
categories: ["scratchpad"]
---






<center>
# This is not a proper blog post yet, just my notes.

Rmarkdown (TODO)
</center>


Before you begin, check out a very nice cheatsheet available at 
https://www.rstudio.com/wp-content/uploads/2016/03/rmarkdown-cheatsheet-2.0.pdf

Vim key mappings are rather intuitive, pretty similar to basic r filetype mappings:
<F2> open up a console window,
<F5> rmarkdown::render()s this file and shows the result in your favourite browser. 

In general, Rmarkdown is just markdown with a possibility to add R chunks of code and execute them. Which is pretty nice, actually. Some more informationon http://rmarkdown.rstudio.com/authoring_basics.html

***

By the way, it would be great if you could have separate keys for rendering and showing the results, like <F5> and <F6>, for example.

And if there was a snippet 
https://github.com/honza/vim-snippets
for rmarkdown so you didn't have to make up your own description block every single time you create a new file. Yeah, you'd love it.

I am sorry but I am too lazy to add these features right now :) Good luck, anyways.



```r
x <- 10
print(x)
```

```
## [1] 10
```

# Some title
just the same function to create a title as in basic markdown

and some plot

![plot of chunk unnamed-chunk-2](./articles/figures/file/unnamed-chunk-2-1.png)

You can also execute bash and probably some other languages as well. That's pretty impressive.

```bash
ls
echo "some random text"
```

```
## aws.md
## caffee.md
## cassandra.md
## classesS4.md
## classesS4.Rmd
## data.table.md
## data.table.Rmd
## debugging.md
## debugging.Rmd
## decision_trees.md
## decision_trees.Rmd
## dimensionality.md
## dimensionality.Rmd
## featuretools.md
## figures
## file.Rmd
## ggplot2.md
## ggplot2.Rmd
## git.md
## hugo.md
## keras.md
## learning_tensorflow.md
## learning_tensorflow.Rmd
## machine_learning_problems.md
## marathon.md
## ml.md
## ml.Rmd
## mtcars.csv
## packages.md
## pandas.md
## passing_arguments.md
## redis.md
## RMariaDB.md
## RMariaDB.Rmd
## rstanarm.md
## sample_data.csv
## shiny.md
## sqldf.md
## sqldf.Rmd
## stringr.md
## stringr.Rmd
## tensorflow.md
## theano.md
## tidyverse.md
## tidyverse.Rmd
## useful_processing.md
## useful_processing.Rmd
## some random text
```
