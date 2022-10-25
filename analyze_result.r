library(tidyverse)
binary_feature = read_csv("C:/Users/huten/Desktop/dfm/result/binary_feature.csv")
ternary_feature = read_csv("C:/Users/huten/Desktop/dfm/result/ternary_feature.csv")
induced_ppmi = read_csv("C:/Users/huten/Desktop/dfm/result/induced_ppmi_class.csv")
induced_pmi = read_csv("C:/Users/huten/Desktop/dfm/result/induced_pmi_class.csv")


binary_feature %>% filter(dev_loss == min(dev_loss)) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
induced_pmi %>% filter(dev_loss == max(dev_loss)) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)

induced_ppmi %>% filter(lr < 0.1) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

induced_ppmi %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
#

induced_pmi %>% filter(lr < 0.1) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

induced_pmi %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
#
ternary_feature %>% filter(lr < 0.1) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

ternary_feature %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
#

induced_class %>% filter(lr < 0.1) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

induced_class %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)