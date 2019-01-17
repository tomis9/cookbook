'''
# disclaimer: I really enjoy working with vim, but I am not an enemy of anyone
# using Emacs. At the end of the day, this is not a text editor that makes us
# good or bad developers and none of us is a specialist in using vim or emacs,
# as these tools are extremely rich in various functionalities. Your
# productivity depends in 95% on you, i.e. you skill in using the editor, and
# your programming skills and knowledge, but not on the editor itself.

# As usual, I am not sure if the data will give me aneough information to tell
# you an interesting story. In other words, there is a risk that this project
# fails commpletely, i.e. vim will not come up any different to emacs,
# according to reddit users. If these editors do not differ enough, I want to
# know it as soon as possible.

# That is why first I will try to prove the most important concepts, before
# making the full analysis, i.e. if there is a chance that the two editors
# differ from each other. As this is the priority now, I have to postpone a few
# analysis, that I will get back to in the future. In other words, I have to
# make assumptions like:

# * I will work on a small subset of data, hoping that the insights it provides
# me with will still be valid on a larger dataset.

# * I will let the praw package sample the subset, hoping that the sampling
# will not be biased.

# * Let's assume that all the posts concerning vim or emacs are contained is
# subreddits "vim" and "emacs".

# So, these are the assumptions that hopefully I will remove as the analysis
# will be proceeding.

# Wish me luck! ;)


# download our own comments
https://www.pythonforbeginners.com/python-on-the-web/how-to-use-reddit-api-in-python/

# the good one:
http://www.storybench.org/how-to-scrape-reddit-with-python/

# the very good one:
https://praw.readthedocs.io/en/latest/getting_started/quick_start.html

nlp - analysis if people like the editor
lda - check what are the best and worst features of these editors
'''

# sudo pip3 install praw
import praw
import pandas as pd
import os
import json

home = os.environ['HOME']
relative_path = 'cookbook/scratchpad/vim_vs_emacs/creds.json'
reddit_creds_path = os.path.join(home, relative_path)
with open(reddit_creds_path, 'r') as f:
    reddit_creds = json.load(f)


reddit = praw.Reddit(client_id=reddit_creds['client_id'],
                     client_secret=reddit_creds['client_secret'],
                     user_agent='vim_vs_emacs',
                     username=reddit_creds['username'],
                     password=reddit_creds['password'])


def get_data(reddit, subreddit_name, n_posts):
    subreddit = reddit.subreddit(subreddit_name)

    top_subreddit = subreddit.top(limit=n_posts)

    topics_dict = {"title": [], "score": [], "comms_num": [], "body": []}

    for submission in top_subreddit:
        topics_dict["title"].append(submission.title)
        topics_dict["comms_num"].append(submission.num_comments)
        topics_dict["score"].append(submission.score)
        topics_dict["body"].append(submission.selftext)

    topics_data = pd.DataFrame(topics_dict)

    return topics_data


editors = ["Atom", "SublimeText", "vscode", "brackets", "notepadplusplus",
           "vim", "emacs"]

editors_posts = []
for editor in editors:
    editor_posts = get_data(reddit, subreddit_name=editor, n_posts=1000)
    editor_posts['editor'] = editor
    editors_posts.append(editor_posts)


result = pd.concat(editors_posts)
save_path = os.path.join(home, 'cookbook/scratchpad/vim_vs_emacs')
result.to_csv(os.path.join(save_path, 'editors_posts.csv'), index=False)
