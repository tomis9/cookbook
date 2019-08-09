import json
from django.db import models
from django.utils import timezone
# from django.contrib.auth.models import User


class Post(models.Model):
    post_slug = models.SlugField(max_length=250, default='empty', primary_key=True)
    title = models.CharField(max_length=250)
    body = models.TextField(default='')
    date = models.DateTimeField(default=timezone.now)
    draft = models.BooleanField(default=True)
    # categories = models.CharField(max_length=100, default='scratchpad')
    # tags = models.CharField(max_length=100, default='scratchpad')

    def get_absolute_url(self):
        return "/post/" + self.post_slug

    def format_date(self):
        return self.date.strftime("%b %-d, %Y")

    class Meta:
        ordering = ('-date',)

    def __str__(self):
        return self.post_slug


class Category(models.Model):
    category_slug = models.CharField(max_length=250, primary_key=True)
    category_name = models.CharField(max_length=250)

    def get_absolute_url(self):
        return "/list/" + self.category_slug

    def __str__(self):
        return self.category_slug


class CategoryPost(models.Model):
    # TODO foreign key applies to the whole object, e.g. Category, not it's particular column, e.g. primary key
    post_slug = models.ForeignKey(Post, on_delete=models.CASCADE)
    category_slug = models.ForeignKey(Category, on_delete=models.CASCADE)
