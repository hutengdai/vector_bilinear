library(tidyverse)
df = read_csv("C:/Users/huten/Desktop/Projects/DFM/code/DFM/result.csv")
head(df)
df %>% filter(dev_loss == min(dev_loss)) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)
df %>% filter(dev_loss == max(dev_loss)) %>% 
  select(step, batch_size, lr, num_iter, dev_loss)


df %>% filter(lr < 0.1) %>% 
  ggplot(aes(x=step, y = dev_loss,
                  color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 


#
df %>% arrange(dev_loss) %>% 
  select(step, batch_size, lr, num_iter, dev_loss) %>% 
  ggplot(aes(x=step, y = dev_loss,
           color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 

df %>% filter(lr = 0.01) %>% 
  ggplot(aes(x=step, y = dev_loss,
             color = as_factor(lr))) + 
  geom_line() + 
  facet_wrap(~batch_size, scale = "free_x") 


