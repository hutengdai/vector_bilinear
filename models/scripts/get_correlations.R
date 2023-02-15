library(tidyverse)
library(corrr)
library(GGally)
library(ggrepel)

#setwd("E:/git_repos/DFM/models")
setwd("C:/Users/conno/git_repos/vector_bilinear/models")

scores_full <- read_csv("results/overall_scores.csv")

scores_full <- scores_full %>%
  mutate(hayes_phonetic_features = -hayes_phonetic_features,
         hayes_learned_features = -hayes_learned_features,
         bl_continuous_pmi = -bl_continuous_pmi,
         bl_discrete_binary_phonetic = -bl_discrete_binary_phonetic,
         bl_discrete_binary_learned = -bl_discrete_binary_learned,
         bl_continuous_type = -bl_continuous_type,
         bl_continuous_laplace = -bl_continuous_laplace,
         hayes_learned_features_lm = -hayes_learned_features_lm,
         bl_discrete_lm = -bl_discrete_lm,
         bl_continuous_lm = -bl_continuous_lm
  ) %>%
  select(-max_rcc_cov, -max_rcc_kld, -max_sc_kld)

# Overall taus correlations
overall_tau <- scores_full %>% 
  select(-onset) %>%
  group_by() %>%
  nest(data=everything()) %>%
  mutate(
    tau = map(data, correlate, method='kendall')
  ) %>%
  unnest(tau) %>% 
  select(term, likert_rating) %>%
  filter(!is.na(likert_rating)) 

overall_r <- scores_full %>% 
  select(-onset) %>%
  group_by() %>%
  nest(data=everything()) %>%
  mutate(
    tau = map(data, correlate, method='pearson')
  ) %>%
  unnest(tau) %>% 
  select(term, likert_rating) %>%
  filter(!is.na(likert_rating)) 

overall_r %>%
  inner_join(overall_tau, by=c('term')) %>%
  mutate(r=likert_rating.x,
         tau=likert_rating.y) %>%
  select(-likert_rating.x, -likert_rating.y) %>%
  write_csv('results/overall_correlations.csv')

# Grouped correlations
grouped_tau <- scores_full %>% 
  select(-onset) %>%
  group_by(attestedness) %>%
  nest() %>%
  mutate(
    correlations = map(data, correlate, method='kendall')
  ) %>%
  unnest(correlations) %>% 
  select(attestedness, term, likert_rating) %>%
  filter(!is.na(likert_rating))

grouped_r <- scores_full %>% 
  select(-onset) %>%
  group_by(attestedness) %>%
  nest() %>%
  mutate(
    correlations = map(data, correlate, method='pearson')
  ) %>%
  unnest(correlations) %>% 
  select(attestedness, term, likert_rating) %>%
  filter(!is.na(likert_rating))

grouped_r %>%
  inner_join(grouped_tau, by=c('attestedness', 'term')) %>%
  mutate(r=likert_rating.x,
         tau=likert_rating.y) %>%
  select(-likert_rating.x, -likert_rating.y) %>%
  write_csv('results/grouped_correlations.csv')
# 
# plot_scores_full <- scores_full %>%
#   pivot_longer(c(-onset, -attestedness, -likert_rating), names_to="model", values_to="score")
# 
# ggplot(plot_scores_full, aes(likert_rating, score, label=onset)) +
#   geom_point() + 
#   geom_text_repel(size=2) +
#   facet_wrap(~model, scale="free")
# ggsave("models_plot.png")
# 
# plot_scores_full %>%
#   filter(model == 'hayes_phonetic_features' | model == 'bl_continuous_pmi') %>%
#   mutate(model = ifelse(model == 'hayes_phonetic_features', 'H&W - discrete phonetic features', 'Log-bilinear - continuous distributional features')) %>%
#   ggplot(aes(x=likert_rating, y=score, label=onset, color=attestedness)) +
#   geom_point(size=2) +
#   geom_text_repel(size=3.5, show.legend = FALSE) +
#   facet_wrap(~ model, scale="free", nrow=2) +
#   ylab("Score") + 
#   xlab("Likert rating") +
#   scale_color_discrete(name = "Attestedness") +
#   theme_classic() +
#   theme(legend.position = c(0.87, 0.15))
# 
# ggsave("figures/scatterplots.png")
