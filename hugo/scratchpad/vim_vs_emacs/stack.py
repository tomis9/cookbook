import stackexchange
import pandas as pd

so = stackexchange.Site(stackexchange.StackOverflow)


def download_posts(tag, n_posts):
    titles = so.search(tagged=tag, filter="withbody")
    posts = []
    i = 0
    for tit in titles:
        i += 1
        post = (tit.title, tit.score, tit.view_count, tit.body, tit.tags)
        posts.append(post)
        if i == n_posts:
            break
    posts = pd.DataFrame(posts)
    posts.columns = ['title', 'score', 'view_count', 'body', 'tags']
    posts['editor'] = tag
    return posts


editors = ['vim', 'emacs', 'atom-editor', 'visual-studio-code', 'sublimetext']
# ides = ['pycharm', 'rstudio']

posts = []

for tag in editors:
    posts.append(download_posts(tag=tag, n_posts=1000))

result = pd.concat(posts)
result.to_csv('posts_stack.csv', index=False)
