---
title: "pyenv"
date: 2018-08-12T15:32:40+01:00
draft: false
categories: ["python", "DevOps"]
tags: ["draft"]
---

### [pyenv + virtualenv](https://amaral.northwestern.edu/resources/guides/pyenv-tutorial)

After the installation from github, which is nicely described in the link above, all you have to do is to

```{bash}
pyenv virtualenv webservice
pyenv activate webservice
pip install flask
python hello.py
...
pyenv deactivate
```

Clearly, you can choose any other name than `webservice`.

Good job, you have just installed flask on a virtualenv, so the other users will not be bothered by this. Maybe they are using any other version of flask than the one you have just installed?

But we can go even deeper with not bothering other users. You can use docker.
(you can also install a specifi version of python with pyenv, but docker is even more powerful)
