---
title: "pyenv, virtualenv, freeze"
date: 2018-08-12T15:32:40+01:00
draft: false
image: "pyenv.jpg"
categories: ["python", "DevOps"]
tags: ["python", "pyenv", "virtualenv", "DevOps"]
---

## 1. What are pyenv, virtualenv and freeze and why would you use them?

* these three Python packages let you install your favourite Python version with your favourite Python packages' versions on any machine independently to those already installed on the system; you can even store many Python versions and Python packages' versions;

* pyenv let's you install any Python version you like;

* virtualenv let's you install any Python's package version you like;

* freeze informs you what packages and which versions of them are already installed. It's very useful when you want to create the same Python environment on any other machine.

## 2. Installation

Installation is trivial. All you have to do is to clone two repositories from github (just copy-paste the code):

pyenv
```
cd
git clone git://github.com/yyuu/pyenv.git .pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

virtualenv
```
git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
source ~/.bashrc
```

Everything is described in "definitely the best tutorial"; link available in section 4.


## 3. A "Hello World" example

Say you want to create a new project with Python 3.6.0.

```{bash}
pyenv install 3.6.0
```

and you want to write an application which uses numpy and flask. First you have to decide how you're going to call your app. And then create a virtualenv with that name.

```{bash}
pyenv virtualenv 3.6.0 awesome_app
```

In order to install packages for this specific Python version and use it's interpreter, let's activate our brand new environment:

```{bash}
pyenv activate awesome_app
```

As you can see, the prompt has changed. Now you can install any package you want with pip:
```{bash}
pip install --upgrade pip  # this may help at the beginning
pip install flask pytest
```

and check the Python's version if it is actually 3.6.0:
```{bash}
python --version
```

Finally, you can list all the packges that are installed under this Python's version:
```{bash}
pip freeze
```

It is a good practice to keep the list of you packages in a `requirements.txt` file
```{bash}
pip freeze > requirements.txt
```

so you can easily install them with
```{bash}
pip install -r requirements.txt
```

when you download your repo from a remote repository.

When the work is done, type
```{bash}
pyenv deactivate
```

## 4. Easily forgettable commands:

* `pyenv virtualenvs` lists all existing virtualenvs

## 5. Useful links:

* [pyenv + virtualenv - definitely the best tutorial](https://amaral.northwestern.edu/resources/guides/pyenv-tutorial)

