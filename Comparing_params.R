library(dplyr)
library(regclass)
library(tidyverse)
library(stargazer)
library(MASS)

data=read.csv('models_altered.csv', stringsAsFactors = T)

# all character columns to factor:
levels(data$initializer_name)[1:2] <-"NONE"
levels(data$optimizer_name)[1] <- "plain SGD"
data$initializer_yn <- ifelse(data$initializer_name != "NONE", "Y", "N")
colnames(data)
data$learning_rate

# Removing large outliers
data = data[-which(data$min_val_loss > 10000),]

# Linear model to explore hyperparameter effects
mod<-lm(min_val_loss~activation_function+learning_rate+batch_norm+hidden_layers+batch_size +initializer_yn, data=data)
summary(mod)

# Plotting Effects 

# Activation Functions
data %>% 
  ggplot(aes(y = min_val_loss, x = activation_function)) + geom_boxplot()

data %>% 
  ggplot(aes(x = min_val_loss, color = activation_function , fill = activation_function)) + 
  geom_density(alpha = 0.1) + ggthemes::theme_clean() + ggtitle("Minimum Validation Loss by Activation Function")
table(data$activation_function)

# Batch Normalization (Y/N)
data$batch_norm_yn = ifelse(data$batch_norm, 'Y','N')
data %>% 
  ggplot(aes(y = min_val_loss, x = batch_norm_yn)) + geom_boxplot() + geom_jitter() + 
  ggthemes::theme_clean() + ggtitle("Minimum Validation Loss by Batch Norm (N/Y)")
table(data$batch_norm)

# Initializer (N/Y)
data %>% 
  ggplot(aes(y = min_val_loss, x = initializer_yn)) + geom_boxplot() + geom_jitter() + 
  ggthemes::theme_clean() + ggtitle("Minimum Validation Loss by Initializer (N/Y)")
table(data$initializer_yn)

# Optimizer
data %>%
  ggplot(aes(y = min_val_loss, x = optimizer_name , fill = optimizer_name)) + 
  geom_boxplot(alpha = 0.1) + ggthemes::theme_clean() + geom_jitter() + 
  ggtitle("Minimum Validation Loss by Optimizer")
table(data$activation_function)

# Comparing Differences Using AOV and Tukey

# Activation Function
AOV = aov(min_val_loss~activation_function, data = data)
par(las=2, mar = c(5,10,5,5))
TUKEY = TukeyHSD(AOV, conf.level=.95)
plot(TUKEY)

# Optimizer
AOV = aov((min_val_loss)~optimizer_name, data = data[-which(data$optimizer_name == 'nesterov'),])
summary(AOV)
TUKEY = TukeyHSD(AOV, conf.level=.95)
par(las=2, mar = c(5,15,5,5))
plot(TUKEY)
table(data$optimizer_name)

# Batch Norm
AOV = aov(min_val_loss~batch_norm, data = data)
summary(AOV)
table(data$batch_norm)

# Initializer (Y/N)
AOV = aov(min_val_loss~initializer_yn, data = data)
summary(AOV)
table(data$initializer_yn)

