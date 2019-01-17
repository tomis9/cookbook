library(dplyr)
library(readr)

vim <- read_csv("vim.csv")
emacs <- read_csv("emacs.csv")

da <- vim %>% 
  mutate(editor = "vim") %>%
  bind_rows(emacs %>%
            mutate(editor = "emacs"))

library(ggplot2)
library(plotly)


p_score <- da %>%
  group_by(editor) %>%
  filter(!(abs(score - median(score)) > 2 * sd(score))) %>%
  ungroup() %>%
  ggplot(mapping = aes(x = score, color = editor)) +
  geom_density()

p_comms_num <- da %>%
  group_by(editor) %>%
  filter(!(abs(comms_num - median(comms_num)) > 2 * sd(comms_num))) %>%
  ungroup() %>%
  ggplot(mapping = aes(x = comms_num, color = editor)) +
  geom_density()

ggplotly(p_score)
ggplotly(p_comms_num)

# so far we can see that in average there are usually comments under vim topics, comparing to emacs, and vim topics usually gain higher score; in other words, they are upvoted more often. This may suggest that there are more vim users than emacs users.

# Or vim is a hotter subject.

# Let's tokenise our dataset.

# preparing for tokenisation

bodies <- da %>%
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
tidy_frame <- function(frame, column) {
  tokens <- tidytext::unnest_tokens_(frame, "word", column)

  tokens <- tokens %>%
    anti_join(tidytext::stop_words, by = "word")

  return(tokens)
}

tokens <- tidy_frame(bodies, "body")

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
