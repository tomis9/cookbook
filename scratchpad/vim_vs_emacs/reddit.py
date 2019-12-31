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
    print("downloading {} finished".format(editor))


result = pd.concat(editors_posts)
save_path = os.path.join(home, 'cookbook/scratchpad/vim_vs_emacs')
result.to_csv(os.path.join(save_path, 'posts_reddit.csv'), index=False)
