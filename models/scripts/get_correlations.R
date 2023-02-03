library(tidyverse)
library(corrr)
library(GGally)

#setwd("E:/git_repos/DFM/models")
setwd("C:/Users/conno/git_repos/vector_bilinear/models")
scores <- read_csv("results/overall_scores.csv")

scores <- scores %>%
  mutate(hayes_phonetic_features = hayes_phonetic_features * -1,
         hayes_learned_features = hayes_learned_features * -1,
         continuous_pmi = continuous_pmi * -1,
         discrete_binary_phonetic = discrete_binary_phonetic * -1,
         discrete_binary_learned = discrete_binary_learned * -1
  )

# Overall correlations
scores %>% 
  select(-word) %>%
  group_by() %>%
  nest(data=everything()) %>%
  mutate(
    correlations = map(data, correlate, method='kendall')
  ) %>%
  unnest(correlations) %>% 
  select(term, human_response) %>%
  filter(!is.na(human_response)) %>%
  write_csv('results/overall_correlations.csv')

# Grouped correlations
scores %>% 
  select(-word) %>%
  group_by(attestedness) %>%
  nest() %>%
  mutate(
    correlations = map(data, correlate, method='kendall')
  ) %>%
  unnest(correlations) %>% 
  select(attestedness, term, human_response) %>%
  filter(!is.na(human_response)) %>%
  write_csv('results/grouped_correlations.csv')

scores %>%
  select(-word, -attestedness) %>%
  ggpairs() 
