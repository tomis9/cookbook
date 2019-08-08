from django.shortcuts import render
from .models import Post
from django.http import HttpResponse


def view_index(request):
    posts = Post.objects.all()
    return render(request, 'blog/post/index.html', {'posts': posts})


def view_list(request):
    pass


def view_single(request, slug):
    post = Post.objects.get(slug=slug)
    return render(request, 'blog/post/single.html', {'post': post})
