library(tidyverse)
setwd("E:/git_repos/DFM/models")
daland_data <- read_csv("Daland_etal_2011__AverageScores.csv")

daland_data_agg <- daland_data %>%
  group_by(onset, attestedness) %>%
  summarize(rating=mean(likert_rating)) %>%
  write_csv("daland_scores_aggregated.csv")
