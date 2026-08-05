library(ggplot2)
library(ggpubr)

p1 <- ggplot(mtcars, aes(mpg, wt, color = factor(cyl))) + geom_point()
p2 <- ggplot(mtcars, aes(mpg, disp, color = factor(cyl))) + geom_point()

panels <- list(p1, p2, p1, p2, p1, p2)

arrange_with_rows <- function(panels) {
  leg <- ggpubr::get_legend(panels[[1]])
  
  row1 <- ggpubr::ggarrange(plotlist = panels[1:3], ncol = 3, nrow = 1, legend = "none")
  row1 <- ggpubr::annotate_figure(row1, right = ggpubr::text_grob("Mishmar", rot = 270, face = "bold", size = 14))
  
  row2 <- ggpubr::ggarrange(plotlist = panels[4:6], ncol = 3, nrow = 1, legend = "none")
  row2 <- ggpubr::annotate_figure(row2, right = ggpubr::text_grob("Zeelim", rot = 270, face = "bold", size = 14))
  
  ggpubr::ggarrange(row1, row2, leg, ncol = 1, nrow = 3, heights = c(1, 1, 0.1))
}

p <- arrange_with_rows(panels)
ggsave("test.png", p, width=10, height=6)
