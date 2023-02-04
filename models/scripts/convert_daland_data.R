library(tidyverse)
#setwd("E:/git_repos/DFM/models")
setwd("C:/Users/conno/git_repos/vector_bilinear/models/")
daland_data <- read_csv("judgments/Daland_etal_2011__AverageScores.csv")

daland_data_agg <- daland_data %>%
  group_by(onset, attestedness) %>%
  summarize(rating=mean(likert_rating)) %>%
  write_csv("judgments/daland_scores_aggregated.csv")

# Make file with duplicates
results <- read_csv("results/overall_scores.csv") %>%
  mutate(onset=word) %>%
  select(-word, -human_response)

full_daland <- daland_data %>%
  select(onset, likert_rating)
  
full_results <- results %>% 
  inner_join(full_daland, by="onset") %>%
  relocate(onset, .before = attestedness) %>%
  relocate(likert_rating, .after=attestedness)

write_csv(full_results, "results/overall_scores_full.csv")
