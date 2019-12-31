---
title: "beautiful soup"
date: 2019-10-11T19:38:20+02:00
draft: false
categories: ["Python"]
---

## 1. What is beautiful soup and why would you use it?

- it's a web scraping Python package

- sometimes your data is allocated on various pages on the internet. Beautiful Soup turns out to be super-helpful in automated collecting of this sort of data.

btw. I love the name. It sounds so randomly.

## 2. The basics

Beautiful Soup can "understand" html code, which you download from the internet using `requests` module:

```{python}
import requests
html = requests.get('https://google.com')
```

unless you look down on mainstream packages and prefer to use something exotic. Then you provide the html to bs in the following way:

```{python}
from bs4 import BeautifulSoup
soup = BeautifulSoup(product_page.content, 'html.parser')
```

and from now on you will be able to use all the goodness of bs:

- `.find` - to find the fist occurrence of an object of particular tag, id or class

```{python}
class_book = soup.find('div', {'class': 'book'})
```

- `.findAll` - all occurrences of a particular tag, id or class,

- `.prettify` - prints html nicely formatted and easy to read,

- `.children` - returns an iterator of alee the children of this tag,

- `.get_text` - returns the text inside of a tag. Useful when retrieving text from links.

You can also treat some tags as dictionaries, e.g.:

```{python}
a = page.find("div", {"class": "menu"}).find('a')
url = a['href']
text = a.get_text()
```
