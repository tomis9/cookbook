from stackapi import StackAPI

SITE = StackAPI('stackoverflow')
posts = SITE.fetch('posts')

import stackexchange

so = stackexchange.Site(stackexchange.StackOverflow)

u = so.user(41981)
print(u.reputation.format())
print(u.answers.count, 'answers')

tits = so.search(tagged='vim')

# https://api.stackexchange.com/docs/advanced-search#order=desc&sort=activity&tagged=vim&filter=default&site=stackoverflow&run=true
l = []
i = 0
for tit in tits:
    i += 1
    l.append(('%8d %s %d' % (tit.id, tit.title, tit.score)))
    if i == 1000:
        break
