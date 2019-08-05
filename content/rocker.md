---
title: "rocker"
date: 2018-12-16T15:47:35+01:00
draft: false
categories: ["R", "DevOps"]
tags: ["R", "docker"]
---

## 1. What is rocker and why would you use it?

* `rocker` is [docker](http://tomis9.com/docker) container specially prepared for working with R programming language;

* it is useful if your R model is a part of a microservice system based on docker containers;

* you can run R/shiny-server/rstudio-server on any machine you want. The only requirement is docker.

## 2. Rocker versions

[An official site](https://hub.docker.com/u/rocker/) of rocker proposes a few versions of r images. The most interesting ones:

* [rstudio](https://hub.docker.com/r/rocker/rstudio) with rstudio server installed, to which you can connect with your favourite Internet browser;

* [shiny](https://hub.docker.com/r/rocker/shiny) with shiny server installed, where you can easily publish your shiny applications. This may be particularly useful for Windows Server users, who may find it difficult to install a shiny server directly;

* [r-ver](https://hub.docker.com/r/rocker/r-ver) with, well, basic R version installed.

## 3. Example of Dockerfile

A nice tutorial on writing a Dockerfile you will [here](http://tomis9.com/docker). As `rocker`'s Dockerfile has it's specific quirks, I present it in a separate blog post.

```
FROM rocker/r-ver:3.5.1

RUN Rscript -e "install.packages('futile.logger')"

RUN R -e "install.packages('devtools')"

RUN install2.r -e bookdown rticles

CMD R
```

We used 3 different methods to install packages. All of them all synonyms, so you can use whichever you like. Probably `install2.r` has the cleanest syntax.

The you can build your image with:

```
docker build -t my_rocker:0.1 .
```

And you're good to go!

## 4. Useful links

* [pretty much this tutorial, but much better and written by someone else](http://ropenscilabs.github.io/r-docker-tutorial/);

*hah, that's funny. The guy who wrote the tutorial above started the first lesson with the words:*

>What is Docker and Why should I use it?

*so in the same manner as I start every blog post.*

## 5. Subjects still to cover

* packrat (TODO)
