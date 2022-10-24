library(tidyverse)
induced_class = read_csv("C:/Users/huten/Desktop/dfm/result/result2022-10-24-14-05-34.csv")
binary_feature = read_csv("C:/Users/huten/Desktop/dfm/result/result.csv")

df = binary_feature
head(df)
df %>% filter(dev_loss == min(dev_loss)) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
df %>% filter(dev_loss == max(dev_loss)) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)


binary_feature %>% filter(lr < 0.1) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

binary_feature %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
#

induced_class %>% filter(lr < 0.1) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

induced_class %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)