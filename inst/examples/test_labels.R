library(ggplot2)
library(ggpubr)

p1 <- ggplot(mtcars, aes(mpg, wt)) + geom_point() + ggtitle("Plot 1")
p2 <- ggplot(mtcars, aes(mpg, wt)) + geom_point() + ggtitle("Plot 2")

p_arr <- ggarrange(p1, p2, labels = c("(a)", "(b)"), label.y = 0.9)

ggsave("test_labels_ggarrange.jpg", p_arr, width=8, height=4)
