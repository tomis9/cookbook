import os
import re
import json
import slugify
import pypandoc
# import logging
from datetime import datetime
import shutil
import textwrap
import readtime

from blog.models import Post, Category, CategoryPost


class Article:

    meta_fields = {'title', 'date', 'draft', 'categories', 'tags'}

    post_fields = {'title', 'date', 'body', 'draft', 'post_slug', 'read_time'}
    category_post_fields = {'post_slug', 'category_slug'}

    draft_dict = {'false': False, 'true': True}

    rmd_meta = textwrap.dedent('''
        ```{r setup, include = FALSE}
        knitr::opts_chunk$set(
        fig.path = "./media/%s/"
        )
        ```
        ''')

    def __init__(self, src_path):
        self.src_path = src_path

        # TODO check which of these are really used
        self.file = os.path.basename(src_path)
        self.file_name, self.file_type = os.path.splitext(self.file)
        self.file_md = self.file_name + ".md"

    def copy_file(self, dest_dir):
        # TODO trycatch on dest_dir if exists
        self.dest_dir = dest_dir
        self.dest_path = os.path.join(dest_dir, self.file)
        # TODO check md5 sum, maybe no need to update this file; keep md5 of
        # Rmd or md file in Post model - create a new field file_md5
        # TODO check if object is in database
        shutil.copy(self.src_path, dest_dir)
        # TODO logging
        print("copied", self.src_path)

    def parse_article(self):

        # import pdb; pdb.set_trace()

        if self.file_type == ".Rmd":
            self._knit_rmd()

        article = dict()
        # TODO trycatch if path_to is set
        file = os.path.join(self.dest_dir, self.file_md)

        with open(file, 'r') as f:
            raw_file = f.read()

        article['body'] = self._prepare_body(raw_file)
        read_time = readtime.of_html(article['body'])
        article['read_time'] = read_time.minutes

        lines = raw_file.splitlines()
        for field in self.meta_fields:
            line = [line for line in lines if line.startswith(field + ":")]
            to_remove_list = [field, '"', '\[', '\]', ':']
            to_remove = "|".join(to_remove_list)
            if line:
                article[field] = re.sub(to_remove, '', line[0])
                article[field] = article[field].rstrip().lstrip()
            else:
                article[field] = ''

        date_raw = article['date'][:-6] + \
            (article['date'][-6:].replace(":", ""))
        article['date'] = datetime.strptime(date_raw, '%Y-%m-%dT%H%M%S%z')

        article['post_slug'] = slugify.slugify(article['title'])
        article['draft'] = self.draft_dict[article['draft']]

        article['category_slug'] = article['categories'] \
            .replace(', ', ',') \
            .split(',')

        self.article = article

    def _knit_rmd(self):
        self._add_rmd_meta()
        # TODO does plotly work?
        cmd = """
        Rscript -e "knitr::knit('articles/{}', output = 'articles/{}')"
        """.format(self.file, self.file_md)
        os.system(cmd)

    def _add_rmd_meta(self):
        with open(self.dest_path, 'r') as f:
            lines = f.readlines()

        nums = []
        for index, line in enumerate(lines):
            if "---" in line:
                nums.append(index)

        meta_full = self.rmd_meta % self.file_name
        meta_lines = meta_full.split("\n")
        meta_newline = [line + "\n" for line in meta_lines]
        new_text = lines[nums[0]:(nums[1]+1)] + \
            ["\n"] + \
            meta_newline + \
            ["\n"] + \
            lines[(nums[1]+1):]

        with open(self.dest_path, 'w') as f:
            f.writelines(new_text)

    def _prepare_body(self, raw_file):
        body_raw = pypandoc.convert_text(raw_file, 'html', format='md')
        body_lines = body_raw.split('\n')
        line_end = ""
        body = ""
        for line in body_lines:
            body += line
            if "<code" in line:
                line_end = "<br>"
            if "</code>" in line:
                line_end = ""
            body += line_end
        return body

    def save_post_instance(self):
        post = {key: value for key, value in self.article.items()
                if key in self.post_fields}
        post_model = Post(**post)
        post_model.save()
        print("post instance saved")

    def save_category_post_instance(self):
        category_post = {key: value for key, value in self.article.items()
                         if key in self.category_post_fields}
        post_slug = category_post['post_slug']
        for category in category_post['category_slug']:
            vs = dict()
            vs['post_slug'] = Post.objects.get(post_slug=post_slug)
            slug_cat = slugify.slugify(category)
            vs['category_slug'] = Category.objects.get(category_slug=slug_cat)
            category_post_model = CategoryPost(**vs)
            category_post_model.save()

        print("category_post instance saved")


if __name__ == '__main__':
    with open('categories.json', 'r') as f:
        categories = json.load(f)

    for category in categories:
        category_instance = Category(**category)
        category_instance.save()

    HOME = os.environ['HOME']
    path_from = os.path.join(HOME, 'cookbook/content')
    path_to = os.path.join(HOME, 'cookbook/django/articles')

    all_files = os.listdir(path_from)
    files = [os.path.join(path_from, file) for file in all_files
             if file.endswith('.md') or file.endswith('.Rmd')]

    for file in files:
        article = Article(file)
        article.copy_file(path_to)
        article.parse_article()
        article.save_post_instance()
        article.save_category_post_instance()

    def clear_db():
        Post.objects.all().delete()
        Category.objects.all().delete()
        CategoryPost.objects.all().delete()
