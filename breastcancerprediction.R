library(tidyverse)
library(neuralnet)
library(GGally)
library(ggplot2)
library(caret)
library(e1071)

dat.bc <- read_csv(file = "breastcancer.csv") %>%
  na.omit()

ggpairs(dat.bc)

scale01 <- function(x) {
  (x - min(x))/(max(x) - min(x))
}

dat.bc <- dat.bc %>%
  mutate(mean_radius = scale01(mean_radius), mean_texture = scale01(mean_texture),
         mean_perimeter = scale01(mean_perimeter), mean_area = scale01(mean_area),
         mean_smoothness = scale01(mean_smoothness))

sample_size <- floor(0.07 * nrow(dat.bc))
set.seed(222)
index <- sample(seq_len(nrow(dat.bc)), size = sample_size)

train <- dat.bc[index,]
test <- dat.bc[-index,]

nn <- neuralnet(diagnosis ~ mean_radius + mean_texture +
                  mean_perimeter + mean_area + mean_smoothness,
                data = train, linear.output = FALSE, err.fct = "ce", 
                likelihood = TRUE, act.fct = "logistic")

plot(nn, rep = "best")
nn$result.matrix

temp_test <- subset(test, select = c("mean_radius", "mean_texture", "mean_perimeter",
                                     "mean_area", "mean_smoothness"))
head(temp_test)

nn.results <- compute(nn, temp_test)
results <- data.frame(actual = test$diagnosis, prediction = nn.results$net.result)

roundedresults <- sapply(results, round, digits = 0)
roundedresultsdf <- data.frame(roundedresults)
attach(roundedresultsdf)

table <- table(actual,prediction)
confusionMatrix(table)