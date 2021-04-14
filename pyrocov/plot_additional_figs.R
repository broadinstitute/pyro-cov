### Create additional figures for pyro-cov paper
# April 13 2021
# lemieux@broadinstitute.org

library(tidyverse)
library(cowplot)
top_strains <- read_tsv("~/Dropbox/COVID/pyro-cov/paper/strains.tsv")[,1:3]
names(top_strains) <- c("strain", "R_eff", "Cases_per_day")
top_strains <- top_strains %>% mutate(log_samples = log10(Cases_per_day)) 
p1 <- ggplot(top_strains, aes(x = R_eff, y = log_samples, label = strain)) + geom_text(size = 3) + theme_bw() + geom_smooth(method = "lm")
p2 <- ggplot(top_strains, aes(x = R_eff)) + geom_histogram(binwidth = 0.05) + theme_bw()

m1 <- lm(data = top_strains, Cases_per_day ~ R_eff)
summary(m1)

plot_grid(p1, p2, nrow = 1, labels = c("A", "B"))
ggsave("~/Dropbox/COVID/pyro-cov/paper/spectrum_transmissibility.jpg")
