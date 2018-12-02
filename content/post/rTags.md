---
title: "rTags"
date: 2018-11-09T23:01:35+01:00
draft: false
categories: ["R"]
image: "rTags.jpg"
tags: ["R", "vim"]
---

## 1. What are rTags and why would you use them?

`rTags` let you jump directly to the definition of a function under your cursor. Modern IDE's provide this functionality, so why wouldn't you have it in vim?


## 2. How to use them?

You don't necessarily have to read the articles from #3 (unless you want to understand what you are doing). All you have to do is run `:RBuildTags`, Nvim-R will create a `tags` file in your current directory and vim will automatically read this file each time you open any .R file in this directory. As simple as that.

To move into a function's definition, type <kbd>CTRL</kbd>+<kbd>]</kbd>, just like in vim's help files.

Remember to re-run `:RBuildTags` each time you add a new function to your project in order to update the list of tags. If you reload tags quite often, consider adding a special mapping to your .vimrc, e.g.:

```{vimscript}
autocmd FileType r nmap <buffer> <F6> :RBuildTags<CR>
```

## 3. Useful links:

* [A nice detailed article](https://developer.r-project.org/rtags.html)
* [R's documentation of rtags function](https://stat.ethz.ch/R-manual/R-devel/library/utils/html/rtags.html)

and, obviously, [Nvim-R plugin](https://github.com/jalvesaq/Nvim-R) documentation (RBuildTags).
