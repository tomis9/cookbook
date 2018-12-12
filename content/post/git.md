---
title: "git"
date: 2018-12-03T15:01:35+01:00
draft: false
image: "git.jpg"
categories: ["data-engineering", "DevOps"]
tags: ["git", "data-engineering", "DevOps"]
---

## 1. What is git and why would you use it?

* Git is a totally basic program if you think seriously about programming. Seriously.

* It's a version control system, which makes:

    * working on the same project with many people simple;

    * remembers the whole history of the projects, i.e. all it's chages as long as you follow git's discipline

## 2. A "Hello world" example

As you may have noticed, my posts usually contain a section called 'A "Hello World" example', but not this time. There are so many tutorials and books available on the Internet that I am sure you will find something suitable for yourself very quickly.

## 3. '6 commands'

Knowing 6 git commands is a humorous description of a basic knowledge of git. But there is much truth in it: you only need 6 commands to work pretty efficiently with git. These are:

* git add

* git commit

* git clone

* git push/pull

* git merge

* git checkout


If you know all of them, you can update your LinkedIn profile with "knowledge of git". If not - I recommend youtube tutorials, like this one:

<iframe width="853" height="480" src="https://www.youtube.com/embed/HVsySz-h9r4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## 4. Slightly more advanced subjects

#### git rebase

In it's consequences, `git rebase` is equivalent to merge, but there are certain differences:

* rebase changes the order of commits - in merge, they are chronological, in rebase - commits from branch 1 go first, then commits from branch 2;

* in merge, you usually checkout to master and run `git merge dev`, in rebase you checkout to dev and run `git rebase master`;

* DON'T REBASE PUBLIC BRANCHES, unless you want to die in the pain :)

In general, when  you work on a specific project with your colleagues, I recommed using rebase, as chronological order is not that important. Thanks to rebase you can scroll the repo log and see the next functionalities (branches) appearing in order. If you even decide to give them special tags, boy, it really helps to kepp order!

Here are a few links which contain more information about rebasing: [one](https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase), [two](https://benmarshall.me/git-rebase/).
