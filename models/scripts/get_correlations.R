library(tidyverse)
library(corrr)
library(GGally)
library(ggrepel)

setwd("E:/git_repos/DFM/models")
#setwd("C:/Users/conno/git_repos/vector_bilinear/models")

scores_full <- read_csv("results/overall_scores.csv")

scores_full <- scores_full %>%
  mutate(hayes_phonetic_features = hayes_phonetic_features * -1,
         hayes_learned_features = hayes_learned_features * -1,
         bl_continuous_pmi = bl_continuous_pmi * -1,
         bl_discrete_binary_phonetic = bl_discrete_binary_phonetic * -1,
         bl_discrete_binary_learned = bl_discrete_binary_learned * -1
  ) 

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

plot_scores_full <- scores_full %>%
  pivot_longer(c(-onset, -attestedness, -likert_rating), names_to="model", values_to="score")

ggplot(plot_scores_full, aes(likert_rating, score, label=onset)) +
  geom_point() + 
  geom_text_repel(size=2) +
  facet_wrap(~model, scale="free")
ggsave("models_plot.png")

