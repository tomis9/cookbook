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

![plot of chunk unnamed-chunk-2](./media/file/unnamed-chunk-2-1.png)

You can also execute bash and probably some other languages as well. That's pretty impressive.

```bash
ls
echo "some random text"
```

```
## airflow.md
## aws.md
## caffee.md
## cassandra.md
## CinR.md
## classesS4.md
## classesS4.Rmd
## cookbook.md
## data.table.md
## data.table.Rmd
## debugging.md
## debugging.Rmd
## decision_trees.md
## decision_trees.Rmd
## decorators.md
## decorators.Rmd
## dimensionality.md
## dimensionality.Rmd
## django.md
## docker.md
## featuretools.md
## file.md
## file.Rmd
## flask.md
## ggplot2.md
## ggplot2.Rmd
## gitlab_ci.md
## git.md
## hadoop.md
## hugo.md
## javascript.md
## kafka.md
## keras.md
## learning_tensorflow.md
## learning_tensorflow.Rmd
## logging.md
## lubridate.md
## lubridate.Rmd
## machine_learning_problems.md
## marathon.md
## mesos.md
## ml.md
## ml.Rmd
## mtcars.csv
## nlp.md
## nls.md
## nls.Rmd
## packages.md
## pandas.md
## passing_arguments.md
## pyenv.md
## pytorch.md
## Rcpp.md
## Rcpp.Rmd
## redis.md
## regex.md
## reshape2.md
## reshape2.Rmd
## RMariaDB.md
## RMariaDB.Rmd
## rocker.md
## rstanarm.md
## rTags.md
## sample_data.csv
## shiny.md
## spark.md
## sqlalchemy.md
## sqldf.md
## sqldf.Rmd
## stringr.md
## stringr.Rmd
## tensorflow.md
## tensorflow_serving.md
## testing.md
## theano.md
## tidyverse.md
## tidyverse.Rmd
## useful_processing.md
## useful_processing.Rmd
## vagrant.md
## validation.md
## validation.Rmd
## vim_vs_emacs.md
## vim_vs_emacs.Rmd
## some random text
```
