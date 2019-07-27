---
title: "writing a cookbook"
date: 2019-07-26T17:58:33+02:00
draft: false
categories: ["Projects"]
tags: []
---

# 1. Why writing a cookbook/blog?

In 2016, when I decided to become a data scientist, I was overwhlemed by the number of skills I had to possess to start this sort of career. Reading job offers convinced me that I should be a specialist in:

* statistics and mathematics (which I learned at the univerity),

* machine learning and data mining (which are fairly easy to learn by yourself if you have a statistical background),

* computer science skills, including:

    * programming (R, Python, VBA, html + css + javascript, SQL, bash - for the beginning)

    * big data / operating systems / algorithms / Linux - understanding parallel computing and why we even need big data, so to understanf limitations of a local computer (CPU/GPU, RAM, disk and how they depend on each other)

    * software development tools - knowing the programming language is not enough if your statistical model is a part of a bigger system. You sould be familiar with intermediate-level git commands (i.e. which do you prefer? merge or rebase and why?), running your code in a production environment (virtualenv, pyenv, pipenv, docker, virtual machines) with CI/CI pipelines and docker on a production server. Don't forget about TDD, it may save your life once in a while, about logging, so you could debug your code efficiently, and packages, to keep your code in a well-organized, professional way. And documentation.

    * showing your results as an API (flask / django / plumber if you want to work in R) - then you should better read a little about REST APIs, or as a dashboard (flask + html, css, javascript / shiny)

So not only you should be a statistician/mathematician, but also a computer scientist.

Many of the skills above are used rather rarely. You may produce only two dashboards a year, write a bash script only once a month, set up a CI/CD pipeline once a year. It would be cumbersome and daunting to learn these things over and over again and it's normal that you foget them, because as a data scientist you usually learn 20 new skills every six months. In result you do not concentrate on consolidation of knowledge, but on possessing new skills, because there are new super tools that there is currently hype on (aw, there is tensorflow 2.0... and snowflake db!) and you should be at least familiar with them or even test them to see if they are worth implemeting.

This cookbook is my way of struggling with the hurricane of knowledge and skills that come through my head, so they stayed with me for a little longer. The posts I write let me quickly get back on tracks with tools that I haven't been using for a long time (e.g. a year or so).

I recommed this way of gathering knowledge to everyone.

# 2. How to write a cookbook/blog?

In the same way as every project you make. Start with something simple an make it better over time. This is the way I was going:

* writing notes in txt files in a folder on my desktop

* creating a git repository on github with my notes, so I could easily access them from work

* started using [knitr](https://yihui.name/knitr/), which let me embed chunks of code in my notes and even execute them and embed the results - very handy! Works for R, Python and bash.

* publishing my notes on [github pages](https://pages.github.com/)

* writing a frontend for my notes in html and css

* as the number of posts was growing, I found it a little difficult to keep track of them, so I moved to [blogdown and hugo](https://bookdown.org/yihui/blogdown/) and kept publishing on github pages

* publishing post manually required a lot of work, so I wrote a pipeline with [travis-ci](https://travis-ci.org/), which published posts on tomis9.github.io automatically after pushing changes to my git repo;

* when I reached ~50 posts, it was taking at least 12 minutes to build my blog on travis. The most time-consuming part was installing dependencies and packages required to build my posts (code embedded in the posts is executed during the built process on travis and this is where the results come from ;)), so I started using [docker hub](https://hub.docker.com/) to keep all the dependencies already installed in an easy-to-acces place.

* I really wanted to give [AWS](https://aws.amazon.com/) a try, so I moved my blog to Amazon S3 and bought a domain whrough Amazon Route 53.

* by then I was using one of Hugo's free blog templates ([link](https://github.com/orianna-zzo/AllinOne)), but it didn't scale properly on mobile devices, so I wrote [my own](https://github.com/tomis9/random_forest)


In the future I am planning to move this cookbook to django, because I would like to be able to mark a few articles as my favourites and have easy access to them. This will require logging in feature. And keep on writing posts, obviously :)
