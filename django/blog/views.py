from django.shortcuts import render
from .models import Post, CategoryPost, Category
from django.http import HttpResponse


def view_index(request):
    posts = Post.objects.all()
    categories = Category.objects.all()
    category_posts = CategoryPost.objects.all()
    context = {'posts': posts, 'categories': categories,
               'category_posts': category_posts}
    return render(request, 'blog/post/index.html', context)


def view_list(request, category_slug):
    category = Category.objects.get(category_slug=category_slug)
    posts = CategoryPost.objects.filter(category_slug=category)
    for post in posts:
        print(post.post_slug.title)

    return render(request, 'blog/post/list.html',
                  {'category': category, 'posts': posts})


def view_single(request, slug):
    post = Post.objects.get(post_slug=slug)
    return render(request, 'blog/post/single.html', {'post': post})
