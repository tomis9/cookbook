# It took me quite a while to find how to download bodies of the questions
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

# nagetive questions like 'cannot' or 'broken'; positive ones like 'upgrade'
# what I am doing is rather a research
# questions that share the tags, i.e. ['emacs', 'vim']

# or analysis of emacs tags (LDA?), i.e. treat a list of tags as a vector of
# words (vector is ordered, as order matters - the most important tags are
# listed earlier

# what languages are used in which editor
