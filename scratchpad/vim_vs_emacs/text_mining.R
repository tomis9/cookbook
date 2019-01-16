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

library(tidytext)
# https://www.tidytextmining.com/tidytext.html#the-unnest_tokens-function
tokens <- unnest_tokens(bodies, "word", body)

data(stop_words)

tokens <- tokens %>%
  anti_join(stop_words)

tokens %>%
  count(editor, word) %>%
  group_by(editor) %>%
  top_n(n = 10) %>%
  ggplot(mapping = aes(x = word, y = n)) +
  geom_bar(stat = "identity") +
  facet_wrap(~editor, nrow = 2) +
  coord_flip()
 
