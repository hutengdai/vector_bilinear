library(tidyverse)
library(corrr)

setwd("E:/git_repos/DFM/models")
scores <- read_csv("results/overall_scores.csv")

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
  write_csv('results/overall_correlations.csv')

scores %>%
  ggplot(aes(x=human_response, y=smoothed_bigram)) +
  geom_point()

scores %>%
  ggplot(aes(x=human_response, y=-hayes_phonetic_features)) +
  geom_point()

scores %>%
  ggplot(aes(x=human_response, y=-log(hayes_learned_features_1_0))) +
  geom_point()

scores %>%
  ggplot(aes(x=human_response, y=-hayes_learned_features_1_5)) +
  geom_point()

scores %>%
  ggplot(aes(x=human_response, y=-hayes_learned_features_2_5)) +
  geom_point()
