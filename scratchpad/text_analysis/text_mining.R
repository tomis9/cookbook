# https://www.tidytextmining.com/tidytext.html

text <- c("Because I could not stop for Death -",
          "He kindly stopped for me -",
          "The Carriage held but just Ourselves -",
          "and Immortality")

text

library(dplyr)
text_df <- data_frame(line = 1:4, text = text)

text_df

# test mining in R, Julia Silge and David Robinson
# A tibble is a modern class of data frame within R, available in the dplyr and tibble packages, that has a convenient print method, will not convert strings to factors, and does not use row names

library(tidytext)

text_df %>%
  tidytext::unnest_tokens(word, text)

# text - from which column of df, word - to which column of new df
# unnest_tokens works a little bit like unpivot / melt

# another interesting case of using `unnnest_tokens` function, but now ngrams (in this particular case - bigrams) are tokens, not words
  
text_df %>%
  tidytext::unnest_tokens(ngram, text, token = "ngrams", n = 2)

library(janeaustenr)
library(dplyr)
library(stringr)

austen <- austen_books()

# a quick exploratory look at our dataset
austen %>%
  dplyr::group_by(book) %>%
  dplyr::summarise(n = n()) %>%
  dplyr::arrange(-n)
# "Emma" is the longest novel, counted by number of lines

reg <- "^chapter [\\divxlc]"
original_books <- austen_books() %>%
  dplyr::group_by(book) %>%
  dplyr::mutate(
    linenumber = row_number(),
    chapter = cumsum(
      stringr::str_detect(text, stringr::regex(reg, ignore_case = TRUE))
    )
  ) %>%
  ungroup()

# a quick check of what happened
original_books %>%
  filter(str_detect(book, "Sense")) %>%
  tail()


# unpivoting by word, so also removing empty lines
tidy_books <- original_books %>%
  unnest_tokens(word, text)

# an average number of words per line
nrow(tidy_books) / nrow(original_books)

# a list of meaningless words, calld "stop words"
data(stop_words)

# excluding stop words from austen list of words
tidy_books <- tidy_books %>%
  anti_join(stop_words)

# a small example of anti_join, because it does not work exactly as expected, i.e. it is not symmetrical
a <- data_frame(a = rep(letters[1:5], each = 3), b = 1:15)
b <- data_frame(a = letters[c(1:2, 6)], some = 1:3)
a %>% anti_join(b)

tidy_books %>%
  count(word, sort = TRUE)

# the same as above
tidy_books %>%
  group_by(word) %>%
  summarise(n = n()) %>%
  arrange(-n)

library(ggplot2)

tidy_books %>%
  count(word, sort = TRUE) %>%
  filter(n > 600) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip()

# let's download some other books
library(gutenbergr)

# hgwells
hgwells <- gutenberg_download(c(35, 36, 5230, 159))

tidy_hgwells <- hgwells %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)

tidy_hgwells %>%
  count(word, sort = TRUE)

# bronte
bronte <- gutenberg_download(c(1260, 768, 969, 9182, 767))

tidy_bronte <- bronte %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)

tidy_bronte %>%
  count(word, sort = TRUE)


library(tidyr)

frequency <- bind_rows(mutate(tidy_bronte, author = "Brontë Sisters"),
                       mutate(tidy_hgwells, author = "H.G. Wells"),
                       mutate(tidy_books, author = "Jane Austen")) %>%
  mutate(word = str_extract(word, "[a-z']+")) %>%
  count(author, word) %>%
  group_by(author) %>%
  mutate(proportion = n / sum(n)) %>%
  select(-n) %>%
  spread(author, proportion) %>%
  gather(author, proportion, `Brontë Sisters`:`H.G. Wells`)

frequency


library(scales)

# expect a warning about rows with missing values being removed
ggplot(frequency, aes(x = proportion, y = `Jane Austen`, color = abs(`Jane Austen` - proportion))) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  scale_color_gradient(limits = c(0, 0.001), low = "darkslategray4", high = "gray75") +
  facet_wrap(~author, ncol = 2) +
  theme(legend.position = "none") +
  labs(y = "Jane Austen", x = NULL)


cor.test(data = frequency[frequency$author == "Brontë Sisters",],
         ~ proportion + `Jane Austen`)

cor.test(data = frequency[frequency$author == "H.G. Wells",],
         ~ proportion + `Jane Austen`)


# https://www.tidytextmining.com/sentiment.html

sentiments
get_sentiments("afinn")
get_sentiments("bing")
get_sentiments("nrc")

library(janeaustenr)
library(dplyr)
library(stringr)

tidy_books <- austen_books() %>%
  group_by(book) %>%
  mutate(linenumber = row_number(),
         chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]",
                                                 ignore_case = TRUE)))) %>%
  ungroup() %>%
  unnest_tokens(word, text)

nrc_joy <- get_sentiments("nrc") %>% 
  filter(sentiment == "joy")

tidy_books %>%
  filter(book == "Emma") %>%
  inner_join(nrc_joy) %>%
  count(word, sort = TRUE)

library(tidyr)

jane_austen_sentiment <- tidy_books %>%
  inner_join(get_sentiments("bing")) %>%
  count(book, index = linenumber %/% 80, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)


library(ggplot2)

ggplot(jane_austen_sentiment, aes(index, sentiment, fill = book)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~book, ncol = 2, scales = "free_x")


library(wordcloud)

tidy_books %>%
  anti_join(stop_words) %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100))


# https://www.tidytextmining.com/tfidf.html

library(dplyr)
library(janeaustenr)
library(tidytext)

book_words <- austen_books() %>%
  unnest_tokens(word, text) %>%
  count(book, word, sort = TRUE) %>%
  ungroup()

total_words <- book_words %>% 
  group_by(book) %>% 
  summarize(total = sum(n))

book_words <- left_join(book_words, total_words)
book_words

book_words <- book_words %>%
  bind_tf_idf(word, book, n)
book_words

# This is the point of tf-idf; it identifies words that are important to one document within a collection of documents.

# collections of documents ist the same as corpus of documents


# https://www.tidytextmining.com/ngrams.html

# https://www.tidytextmining.com/dtm.html

library(tm)

# sudo apt-get install libgsl0-dev
data("AssociatedPress", package = "topicmodels")
AssociatedPress
terms <- Terms(AssociatedPress)

library(dplyr)
library(tidytext)

ap_td <- tidy(AssociatedPress)
ap_td

ap_td %>% count(document)
tail(ap_td)

# https://www.tidytextmining.com/topicmodeling.html

library(topicmodels)
ap_lda <- LDA(AssociatedPress, k = 2, control = list(seed = 1234))
ap_lda

library(tidytext)

ap_topics <- tidy(ap_lda, matrix = "beta")
ap_topics %>% arrange(-beta)


