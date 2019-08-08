import os
import re
import json
import slugify
import pypandoc
# import logging
from datetime import datetime
import shutil
import textwrap

from blog.models import Post, Category, CategoryPost


class Article:

    singular_fields = ['title', 'date', 'draft']
    multiple_fields = ['categories', 'tags']

    draft_dict = {'false': False, 'true': True}

    rmd_meta = textwrap.dedent('''
        ```{r setup, include = FALSE}
        knitr::opts_chunk$set(
        fig.path = "./articles/figures/%s/"
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
        # TODO check md5 sum, maybe no need to update this file
        shutil.copy(self.src_path, dest_dir)
        # TODO logging
        print("copied", self.src_path)

    def prepare_model_instance(self):

        if self.file_type == ".Rmd":
            self._knit_rmd()

        post = dict()
        # TODO trycatch if path_to is set
        file = os.path.join(self.dest_dir, self.file_md)

        with open(file, 'r') as f:
            raw_file = f.read()

            post['body'] = self._prepare_body(raw_file)

            lines = raw_file.splitlines()
            # TODO singular and multiple fields should be a dict
            for field in self.singular_fields:
                line = [line for line in lines if line.startswith(field + ":")]
                to_remove_list = [field, '"', '\[', '\]', ':']
                to_remove = "|".join(to_remove_list)
                if line:
                    post[field] = re.sub(to_remove, '', line[0]) \
                        .rstrip() \
                        .lstrip()
                else:
                    post[field] = ''

                if field in self.multiple_fields:
                    post[field] = json.dumps(post[field].split(','))

            date_raw = post['date'][:-6] + (post['date'][-6:].replace(":", ""))
            date = datetime.strptime(date_raw, '%Y-%m-%dT%H%M%S%z')

            post['date'] = date

            slug = slugify.slugify(post['title'])
            post['slug'] = slug
            post['draft'] = self.draft_dict[post['draft']]

            self.post_model = Post(**post)

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

    def save_model_instance(self):
        self.post_model.save()
        print("model instance saved")

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

    def _knit_rmd(self):
        self._add_rmd_meta()
        cmd = """
        Rscript -e "knitr::knit('articles/{}', output = 'articles/{}')"
        """.format(self.file, self.file_md)
        os.system(cmd)


if __name__ == '__main__':
    HOME = os.environ['HOME']
    path_from = os.path.join(HOME, 'files/content')
    path_to = os.path.join(HOME, 'projects/files/articles')

    all_files = os.listdir(path_from)
    files = [os.path.join(path_from, file) for file in all_files
            if file.endswith('.md') or file.endswith('.Rmd')]

    for file in files:
        article = Article(file)
        article.copy_file(path_to)
        article.prepare_model_instance()
        article.save_model_instance()


categories = [{'category': 'python', 'category_name': 'Python'},
              {'category': 'r', 'category_name': 'R'},
              {'category': 'machine-learning', 'category_name': 'Machine learning'},
              {'category': 'data-engineering', 'category_name': 'Data engineering'},
              {'category': 'devops', 'category_name': 'DevOps'},
              {'category': 'projects', 'category_name': 'Projects'},
              {'category': 'scratchpad', 'category_name': 'Scratchpad'}]
for category in categories:
    category_instance = Category(**category)
    category_instance.save()
