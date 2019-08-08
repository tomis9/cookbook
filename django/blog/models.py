import json
from django.db import models
from django.utils import timezone
# from django.contrib.auth.models import User

# TODO category mapping should be in a separate table consisting of category_id and slug;
#  looks like the parser will be slightly more complicated


class Post(models.Model):
    # TODO primary key on slug
    slug = models.SlugField(max_length=250, default='empty', primary_key=True)
    title = models.CharField(max_length=250)
    body = models.TextField(default='')
    date = models.DateTimeField(default=timezone.now)
    draft = models.BooleanField(default=True)
    categories = models.CharField(max_length=100, default='scratchpad')
    tags = models.CharField(max_length=100, default='scratchpad')

    def get_absolute_url(self):
        return "post/" + self.slug

    def format_date(self):
        return self.date.strftime("%b %-d, %Y")

    def unpack_categories(self):
        return json.loads(self.categories)

    class Meta:
        ordering = ('-date',)

    def __str__(self):
        return self.slug


class Category(models.Model):
    category = models.CharField(max_length=250, primary_key=True)
    category_name = models.CharField(max_length=250)

    def __str__(self):
        return self.category


class CategoryPost(models.Model):
    slug = models.ForeignKey(Post, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
