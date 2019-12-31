---
title: "vim as IDE"
date: 2019-08-24T16:30:19+02:00
draft: true
categories: ["scratchpad"]
tags: []
---

So, after almost 3 years of using vim I've finally decided to switch to noevim for the following reasons:

- I am a really *dark* vim user, i.e. I use a lot of plugins

- many plugins were written in the era when vim could not run tasks asynchronously (vim versions up to vim 8) and even though they are usually adapted to vim 8, they sometimes don't work properly

- in my opinion vim works better for non-developers, like sysadmins or devops engineers. As I usually write code only on my laptop, I don't take advantage of the fact that vim is avalable on every linux machine, so I use it only locally, so I install many plugins (why not?), so eventually vim's brother (nvim) will be just like another vim;s plugin.

- there are many interesting tutorials on how to set up your python IDE in vim, and they all suggest using neovim: [1](https://yufanlu.net/2018/09/03/neovim-python/), [2](https://jdhao.github.io/2018/12/24/centos_nvim_install_use_guide_en/), [3](https://medium.com/@hanspinckaers/setting-up-vim-as-an-ide-for-python-773722142d1d)

- neovim *is* vim, so I am going to create an alias for nvim `export vim=nvim`. Except for better performance, I should't notice any difference.

Why am I explaining myself in so many words, as if I was guilty? I really wanted to stay with pure vim, but I just can't. I'm sorry, vim. It's not *you*, it's *me*.

```{bash}
curl -fLo ~/.local/share/nvim/site/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```
