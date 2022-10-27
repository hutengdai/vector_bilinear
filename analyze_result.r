library(tidyverse)
binary_feature = read_csv("C:/Users/huten/Desktop/dfm/result/binary_feature.csv")
ternary_feature = read_csv("C:/Users/huten/Desktop/dfm/result/ternary_feature.csv")
induced_ppmi = read_csv("C:/Users/huten/Desktop/dfm/result/induced_ppmi_class.csv")
induced_pmi = read_csv("C:/Users/huten/Desktop/dfm/result/induced_pmi_class.csv")


binary_feature %>% filter(dev_loss == min(dev_loss)) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
induced_pmi %>% filter(dev_loss == max(dev_loss)) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)

# Binary features (problematic)
# 32 0.01 2.58
# 64 0.01 2.59 
# 64 0.001 2.59 [GOOD] track dev_loss in rerun

binary_feature %>% filter(lr < 0.01) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

a <- binary_feature %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
head(a, 25)

# Ternary features (allow negative values but 
# less phonological features)
# 32 0.001 1.57 
# 512 0.01 1.57
# 64 0.001 1.57 [GOOD]
ternary_feature %>% filter(lr < 0.01) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

a <- ternary_feature %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
head(a, 20)

# induced ppmi 
# 64 0.01 2.63 [GOOD]
# 32 0.01 2.59
induced_ppmi %>% filter(lr < 0.1) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

a <- induced_ppmi %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
head(a,20)

# induced pmi (and ternary features both allow 
# negative values)
# 512 0.01 1.58
# 32 0.01 1.58
# 64 0.001 1.58 [GOOD]

induced_pmi %>% filter(lr < 0.01) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

a <- induced_pmi %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
head(a,20)

