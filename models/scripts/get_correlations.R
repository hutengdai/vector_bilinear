library(tidyverse)
library(corrr)
library(GGally)
library(ggrepel)

#setwd("E:/git_repos/DFM/models")
setwd("C:/Users/conno/git_repos/vector_bilinear/models")
scores <- read_csv("results/overall_scores_aggregated.csv")

scores <- scores %>%
  mutate(hayes_phonetic_features = hayes_phonetic_features * -1,
         hayes_learned_features = hayes_learned_features * -1,
         continuous_pmi = continuous_pmi * -1,
         discrete_binary_phonetic = discrete_binary_phonetic * -1,
         discrete_binary_learned = discrete_binary_learned * -1
  )

scores_full <- read_csv("results/overall_scores.csv")

scores_full <- scores_full %>%
  mutate(hayes_phonetic_features = hayes_phonetic_features * -1,
         hayes_learned_features = hayes_learned_features * -1,
         continuous_pmi = continuous_pmi * -1,
         discrete_binary_phonetic = discrete_binary_phonetic * -1,
         discrete_binary_learned = discrete_binary_learned * -1
  ) 

# Overall correlations
scores %>% 
  select(-onset) %>%
  group_by() %>%
  nest(data=everything()) %>%
  mutate(
    correlations = map(data, correlate, method='kendall')
  ) %>%
  unnest(correlations) %>% 
  select(term, likert_rating) %>%
  filter(!is.na(likert_rating)) %>%
  write_csv('results/overall_correlations_aggregated.csv')

# Grouped correlations
scores %>% 
  select(-onset) %>%
  group_by(attestedness) %>%
  nest() %>%
  mutate(
    correlations = map(data, correlate, method='kendall')
  ) %>%
  unnest(correlations) %>% 
  select(attestedness, term, likert_rating) %>%
  filter(!is.na(likert_rating)) %>%
  write_csv('results/grouped_correlations_aggregated.csv')

# Overall correlations
scores_full %>% 
  select(-onset) %>%
  group_by() %>%
  nest(data=everything()) %>%
  mutate(
    correlations = map(data, correlate, method='kendall')
  ) %>%
  unnest(correlations) %>% 
  select(term, likert_rating) %>%
  filter(!is.na(likert_rating)) %>%
  write_csv('results/overall_correlations.csv')

# Grouped correlations
scores_full %>% 
  select(-onset) %>%
  group_by(attestedness) %>%
  nest() %>%
  mutate(
    correlations = map(data, correlate, method='kendall')
  ) %>%
  unnest(correlations) %>% 
  select(attestedness, term, likert_rating) %>%
  filter(!is.na(likert_rating)) %>%
  write_csv('results/grouped_correlations.csv')



plot_scores <- scores %>%
  pivot_longer(c(-onset, -attestedness, -likert_rating), names_to="model", values_to="score")

plot_scores_full <- scores_full %>%
  pivot_longer(c(-onset, -attestedness, -likert_rating), names_to="model", values_to="score")

ggplot(plot_scores, aes(likert_rating, score, label=onset)) +
  geom_point() +
  geom_text_repel(size=2) +
  facet_wrap(~model, scale="free")
ggsave("averaged_models_plot.png")

ggplot(plot_scores_full, aes(likert_rating, score, label=onset)) +
  geom_point() + 
  geom_text_repel(size=2) +
  facet_wrap(~model, scale="free")
ggsave("models_plot.png")

