library(dplyr)
library(readr)

editors_posts <- read_csv("editors_posts.csv")

library(ggplot2)
library(plotly)

editors_posts <- editors_posts %>%
  group_by(editor) %>%
  mutate(n = n()) %>%
  filter(n > 500) %>%
  ungroup()

p_score <- editors_posts %>%
  group_by(editor) %>%
  filter(!(abs(score - median(score)) > 2 * sd(score))) %>%
  ungroup() %>%
  ggplot(mapping = aes(x = score, color = editor)) +
  geom_density()

p_comms_num <- editors_posts %>%
  group_by(editor) %>%
  filter(!(abs(comms_num - median(comms_num)) > 2 * sd(comms_num))) %>%
  ungroup() %>%
  ggplot(mapping = aes(x = comms_num, color = editor)) +
  geom_density()

ggplotly(p_score)
ggplotly(p_comms_num)

# so far we can see that in average there are usually comments under vim topics, comparing to emacs, and vim topics usually gain higher score; in other words, they are upvoted more often. This may suggest that there are more vim users than emacs users.
# or people ask on stackoverflow instead of reddit.
# https://github.com/lucjon/Py-StackExchange

# Or vim is a hotter subject.

# Let's tokenise our dataset.

# preparing for tokenisation

bodies <- editors_posts %>%
  dplyr::filter(!is.na(body)) %>%
  group_by(editor) %>%
  mutate(post_num = 1:n()) %>%
  ungroup() %>%
  select(editor, post_num, body)

# quick check if our sample dataset is still balanced
bodies %>%
  count(editor)


# all the quotes/disclaimers are my comments regarding my workflow or personal opinions, not the analysis itself
#> all the calculations above could have been performed on spark using sparklyr, but I didn't do it, as this would be a huge overkill for a dataset that small. However, if I obtain interesting results, I will move to (sparklyr)[https://spark.rstudio.com/guides/textmining/] and maybe run calclulations on AWS. For now let's use tidytext and work locally, while keeping the app pretty easy to scale (e.g. encapsulate into functions these calculations, which can be replaced with sparklyr int the nearest future).

#######################################
# a quick exploratory analysis

tokens <- tidytext::unnest_tokens(bodies, "word", body)

tokens <- tokens %>%
  anti_join(tidytext::stop_words, by = "word")

tokens <- tokens %>%
  filter(!word %in% c("vim", "emacs"))

# https://www.tidytextmining.com/tidytext.html#the-unnest_tokens-function

tokens %>%
  count(editor, word) %>%
  group_by(editor) %>%
  top_n(n = 10) %>%
  ggplot(mapping = aes(x = word, y = n)) +
  geom_bar(stat = "identity") +
  facet_wrap(~editor, nrow = 2) +
  coord_flip()
 
library(tidyr)
library(ggplot2)
# library(tidyr)
p_t <- tokens %>%
  count(editor, word) %>%
  select(editor, word, n_occur = n) %>%
  group_by(editor) %>%
  mutate(freq = n_occur / sum(n_occur)) %>%
  select(editor, word, freq) %>%
  spread(editor, freq) %>%
  ggplot(aes(x = vim, y = emacs)) +
  geom_jitter() +
  geom_text(aes(label = word))
ggplotly(p_t)

# you can zoom the plot easily

# package vs plugin - emacs users clearly use packages, while vim users - plugins
# vim plugins are stored on github
# undoing seems to be the case among emacs users
# vim commands (command mode) is still complicated and user-unfriendly
# vim 8, which is relatively new, makes multi-threaded computations much easier
# emacs users are more aware of documentation ;)

# a dictionary of synonyms could be useful - we could cluster the words


#######################################
# sentiment analysis
# ithey do, t is very difficult to tell which text editor is better, as there are very few people who know well both of them. Even if they do, they are probably very... unusual people with unusual hobbys or needs, and generalization made on observing them would be doubtful.
# But, what we can do, is measuring programmers' satisfaction among vim and emacs users.

# sentiment analysis using lstm and tensorflow
# https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow

tokens %>%
  inner_join(tidytext::get_sentiments("afinn")) %>%
  group_by(editor, post_num) %>%
  summarise(mean_score = mean(score)) %>%
  ungroup() %>%
  ggplot(aes(x = mean_score, color = editor)) +
  geom_density()

# it seems like vim users are happier. We could still perform this analysis with sparklyr.
  

library(wordcloud)

tokens %>%
  filter(editor == "vim") %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100))

# wider analysis: specific words for emacs and vim comparing to atom (Atom), sublime text (SublimeText), visual studio code (vscode), Brackets (brackets), notepad++ (notepadplusplus)

# what's next? tf-idf  https://www.tidytextmining.com/tfidf.html

# tf-itf

library(tidytext)

editor_words <- tokens %>%
  count(editor, word) %>%
  arrange(editor, -n) %>%
  bind_tf_idf(word, editor, n) %>%
  arrange(-tf_idf)

editor_words %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>% 
  group_by(editor) %>% 
  top_n(15) %>% 
  ungroup %>%
  ggplot(aes(word, tf_idf, fill = editor)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~editor, ncol = 2, scales = "free") +
  coord_flip()

# what's next? bi-gram analysis https://www.tidytextmining.com/ngrams.html

# http://analytics-magazine.org/text-analytics-deriving-value-from-tweets-blog-posts-and-call-center-logs/
# https://www.researchgate.net/publication/4361788_Using_text_mining_to_analyze_user_forums/download

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.720.6047&rep=rep1&type=pdf


# stackoverflow -----------
library(tidyverse)
library(readr)
library(ggplot2)

editors_posts <- read_csv("posts_stack.csv")

editors_posts %>%
  filter(score < quantile(score, 0.9)) %>%
  ggplot(mapping = aes(x = score, color = editor)) +
  geom_density(bw = 0.5)

# tags analysis
# users of some languages may be more frustrated than the others'... I felt pretty frustrated when I had to code in VBA. As some editors may have particuralry well-written plugins for specific languages, we cannot say the our posts sample is IID. Let's analise the dependencies.
  
tags <- editors_posts %>%
  select(editor, tags) %>%
  unnest(tag = strsplit(tags, ",")) %>%
  mutate(tag = gsub("[]\\[\\' ]", "", tag)) %>%
  select(editor, tag)

# it looks like pretty much everybody codes in python, but some javascript users use atom, sublime text and vs-code.
library(wordcloud)

editor_synonyms <- c("vi", "sublimetext2", "sublimetext3")

tags %>%
  filter(editor != tag) %>%
  filter(!tag %in% editor_synonyms) %>%
  filter(editor == "visual-studio-code") %>%
  count(tag) %>%
  with(wordcloud(tag, n, max.words = 50))

# sentiment analysis
bodies <- editors_posts %>%
  dplyr::filter(!is.na(body)) %>%
  group_by(editor) %>%
  mutate(post_num = 1:n()) %>%
  ungroup() %>%
  select(editor, post_num, body) %>%
  mutate(body = gsub("<.*?>", "", body))


# all the quotes/disclaimers are my comments regarding my workflow or personal opinions, not the analysis itself
#> all the calculations above could have been performed on spark using sparklyr, but I didn't do it, as this would be a huge overkill for a dataset that small. However, if I obtain interesting results, I will move to (sparklyr)[https://spark.rstudio.com/guides/textmining/] and maybe run calclulations on AWS. For now let's use tidytext and work locally, while keeping the app pretty easy to scale (e.g. encapsulate into functions these calculations, which can be replaced with sparklyr int the nearest future).

#######################################
# a quick exploratory analysis

tokens <- tidytext::unnest_tokens(bodies, "word", body)

tokens <- tokens %>%
  anti_join(tidytext::stop_words, by = "word")

tokens %>%
  inner_join(tidytext::get_sentiments("afinn")) %>%
  group_by(editor, post_num) %>%
  summarise(mean_score = mean(score)) %>%
  ungroup() %>%
  ggplot(aes(x = mean_score, color = editor)) +
  geom_density()

bads <- tokens %>%
  inner_join(tidytext::get_sentiments("afinn")) %>%
  group_by(editor, post_num) %>%
  summarise(mean_score = mean(score)) %>%
  filter(mean_score == -2) %>%
  inner_join(bodies)

as.character(bads[1, "body"])
as.character(bads[2, "body"])
as.character(bads[3, "body"])

goods <- tokens %>%
  inner_join(tidytext::get_sentiments("afinn")) %>%
  group_by(editor, post_num) %>%
  summarise(mean_score = mean(score)) %>%
  filter(mean_score == 2) %>%
  inner_join(bodies)

as.character(goods[1, "body"])
as.character(goods[2, "body"])
as.character(goods[3, "body"])

# does the sentiment analysis even work here? A little bit, I would say. The posts marked as "positive" usually concern a user who wants to add a new functionality to make his/her experience with the text editor even better, e.g. they have seen this functionality working in another editor, like the one they used before, and want to make it running in this one. On the other hand, the most "negative" posts include the word error, so the user tried to use the functionality which *should* be working, but an error message pops up. As you can see, whether the post is positive or negative, may depend solely on the expectations and mood of the user. As these factors are rather uniform across the editors, they are independent to the quality of the editor, so we can use some statistics to check if our current results are statistically significant.

# here I could use mcmc to create distributions of means of scores

# https://nicercode.github.io/guides/mcmc/ - another good question, as if I was reading my own blog post :)

tokens %>%
  inner_join(tidytext::get_sentiments("afinn")) %>%
  group_by(editor, post_num) %>%
  summarise(mean_score = mean(score)) %>%
  group_by(editor) %>%
  summarise(mean_editor = mean(mean_score))

s <- sample_n(bodies, 1000)

tokens <- tidytext::unnest_tokens(s, "word", body)

tokens <- tokens %>%
  anti_join(tidytext::stop_words, by = "word")

tokens %>%
  inner_join(tidytext::get_sentiments("afinn")) %>%
  group_by(editor, post_num) %>%
  summarise(mean_score = mean(score)) %>%
  group_by(editor) %>%
  summarise(mean_editor = mean(mean_score))

