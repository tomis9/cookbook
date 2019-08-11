from django.shortcuts import render
from .models import Post, CategoryPost, Category
from django.http import HttpResponse
from django.db.models import Count

# TODO providing category and category_posts by decorator
def view_index(request):
    posts = Post.objects.all()
    categories = Category.objects.all()
    category_posts = CategoryPost.objects.all()
    cat_nums = CategoryPost.objects.values('category_slug').annotate(dcount=Count('category_slug'))
    print(cat_nums)
    context = {'posts': posts, 'categories': categories,
               'category_posts': category_posts, 'cat_nums': cat_nums}
    return render(request, 'blog/post/index.html', context)


def view_list(request, category_slug):
    category = Category.objects.get(category_slug=category_slug)
    posts = CategoryPost.objects.filter(category_slug=category)
    categories = Category.objects.all()
    category_posts = CategoryPost.objects.all()
    context = {'category': category, 'posts': posts, 'categories': categories,
               'category_posts': category_posts}
    return render(request, 'blog/post/list.html', context)


def view_single(request, slug):
    post = Post.objects.get(post_slug=slug)
    categories_specific = CategoryPost.objects.filter(post_slug=slug)
    categories = Category.objects.all()
    category_posts = CategoryPost.objects.all()
    context = {'post': post, 'categories': categories,
               'category_posts': category_posts,
               'categories_specific': categories_specific}
    return render(request, 'blog/post/single.html', context)
