---
title: "Aws"
date: 2019-06-17T17:07:12+02:00
draft: true
categories: ["scratchpad"]
tags: []
---

https://www.youtube.com/watch?v=g9NbuTcos18


A simple project:

- moving my blog to aws

[hosting a static website on aws](https://docs.aws.amazon.com/AmazonS3/latest/dev/website-hosting-custom-domain-walkthrough.html)

- providing special functionalities to my blog, which will:

    - count the number of visitors;

    - display a different content for me and other visitors. I would be recognized by login (I would have to provide logging in flask and maybe possibility for creating logins) or recognising with ip, or with cookies (I am really curious how to do this :))

    - letting users choose their favourite blog posts and keeping them in "favourites" folder;

    - article recommendations based on your browsing history. Which I collect.


https://medium.com/@michal.frystacky/static-site-github-to-s3-770953a90f67


- using awscli

```
aws s3 sync s3://tomis9.com
```
