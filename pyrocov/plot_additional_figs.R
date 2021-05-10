### Create additional figures for pyro-cov paper
# April 13 2021
# lemieux@broadinstitute.org

library(tidyverse)
library(cowplot)
library(gt)
library(ggpubr)

top_strains <- read_tsv("~/Dropbox/COVID/pyro-cov/paper/strains.tsv")#[,1:3]
names(top_strains) <- c("rank" ,"strain", "R_eff", "lower_ci","upper_ci","Cases_per_day", "Cases_total", "birthday")
top_strains <- top_strains %>% mutate(log_samples = log10(Cases_per_day)) 
p1 <- ggplot(top_strains, aes(x = R_eff, y = Cases_per_day, label = strain)) + geom_text(size = 3) + 
  #geom_errorbarh(aes(xmin = lower_ci, xmax = upper_ci)) +
  theme_bw() + 
  #geom_smooth(method = "lm") + 
  labs(x = "Fold Increase in Reproductive Number", y = "Log10(Cases per Day)") + 
  scale_y_log10()
p2 <- ggplot(top_strains, aes(x = R_eff)) + geom_histogram(binwidth = 0.05) + theme_bw() + 
  labs(x = "Fold Increase in Reproductive Number", y = "Count")

p3 <- ggplot(top_strains, aes(y = R_eff, x = birthday, label = strain)) +
  geom_text(size = 3) + theme_bw() + 
  labs( x= "Date of Lineage Emergence", y = "Fold Increase in Reproductive Number") + 
  geom_smooth(method = "lm")

p3
m1 <- lm(data = top_strains, Cases_per_day ~ R_eff)
summary(m1)
m2 <- lm(data = top_strains, R_eff ~ birthday)
summary(m2)

plot_grid(p2, p1,p3, nrow = 1, labels = c("A", "B", "C"))
ggsave("~/Dropbox/COVID/pyro-cov/paper/spectrum_transmissibility.jpg", height = 4, width = 12)



# look at mutation occurance summaries


high_transmit_muts <- top_muts %>% filter(rank < 101)
low_transmit_muts <- top_muts %>% filter(rank > 100)
quantile(low_transmit_muts$num_emergences, c(0.05, 0.5, 0.95))
quantile(high_transmit_muts$num_emergences, c(0.05, 0.5, 0.95))
summary(low_transmit_muts$num_emergences)
summary(high_transmit_muts$num_emergences)
ks.test(low_transmit_muts$num_emergences, high_transmit_muts$num_emergences)

p0.1 <- ggplot(top_muts, aes(x = num_emergences)) + 
  geom_histogram(bins = 10) + 
  scale_x_log10() + 
  theme_bw() + 
  labs(x = "Number of Emergences", title = "Bottom 2132 Most Transmissible Lineages")
p0.1

p0.2 <- ggplot(high_transmit_muts, aes(x = num_emergences)) + 
  geom_histogram(bins = 10) + 
  scale_x_log10() + 
  theme_bw() + 
  labs(x = "Number of Emergences", title = "100 Most Transmissible Mutations")
p0.2
emergence_dist <- rbind(data.frame(num_emergences = high_transmit_muts[c("num_emergences")], Rank = "Top 100"), 
                        data.frame(num_emergences = low_transmit_muts[c("num_emergences")], Rank = "Bottom 2132"))

p0.3 <- ggplot(emergence_dist, aes(x = num_emergences, fill = Rank)) +
  geom_histogram(aes(y = stat(count) / sum(count)),bins = 10) + 
  scale_x_log10() + 
  theme_bw() + 
  labs(x = "Number of Emergences")
p0.3
plot_grid(p0.1, p0.2, labels = c("A", "B"))

ggsave("~/Dropbox/COVID/pyro-cov/paper/convergent_evolution.jpg", height = 4, width = 12)

# construct table 1 
strains_table <- top_strains[,c("rank", "strain", "R_eff", "lower_ci", "upper_ci", "Cases_per_day")]
strains_table <- strains_table %>% mutate(CI = paste(round(lower_ci, 2), round(upper_ci, 2), sep="-")) %>% 
  mutate(R_eff = round(R_eff, 2)) %>% 
  mutate(Cases_per_day = round(Cases_per_day, 0))
strains_table <- strains_table[,c("rank", "strain", "R_eff", "CI", "Cases_per_day")]
names(strains_table) <- c("Rank", "Pango Lineage", "Fold Increase in R", "Delta R_eff CI", "Cases per day")
t1 <- strains_table[1:10,] %>% gt() %>% 
  cols_align(align = "center") %>% tab_style(style = cell_text(weight = "bold"), 
                                             locations = cells_column_labels(columns = everything())
                                             )
t1

# construct table 2

top_muts <- read_tsv("~/Dropbox/COVID/pyro-cov/paper/mutations.tsv")
names(top_muts) <- c("rank" ,"mutation", "mean_sd","mean", "lower_ci","upper_ci","R_eff", "emerged_in")
top_muts <- top_muts %>% mutate(num_emergences = 
                                  sapply(emerged_in, function(x) length(str_split(x, ",")[[1]]))) %>% 
  mutate(CI = paste(round(lower_ci, 2), round(upper_ci, 2), sep="-")) %>% 
  mutate(R_eff = round(R_eff, 2)) %>% 
  mutate(mean = round(mean, 2)) %>% 
  mutate(mean_sd = round(mean_sd, 2))

muts_table <- top_muts[,c("rank", "mutation","mean", "CI", "R_eff", "num_emergences")]
names(muts_table) <- c("Rank","AA Substitution", "mean", "95% CI", "Fold Increase in R", "Number of Lineages")
t2 <- muts_table[1:20,] %>% gt() %>% 
  cols_align(align = "center") %>% tab_style(style = cell_text(weight = "bold"), 
                                             locations = cells_column_labels(columns = everything())
  )
t2

# plot 

top_muts <- top_muts %>% 
  mutate(gene = sapply(top_muts$mutation, function(x) str_split(x, ":")[[1]][1])) %>% 
  mutate(Effect = ifelse(top_muts$R_eff > 1, "Increase Transmissibility", "Decrease Transmissibility")) %>%
  mutate(Rank = ifelse(top_muts$rank < 101, "Top 100", "Bottom 2132"))


p4 <- ggplot(data = subset(top_muts, R_eff > 0 & abs(mean_sd) > quantile(abs(mean_sd), 0.9)), aes(x = gene, y = R_eff)) + 
  geom_violin(draw_quantiles = c(0.5), adjust = 2) + 
  geom_jitter(aes(color = Effect), width = 0.3, height = 0) + 
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 90)) + 
  labs( x = "Gene", y = "Fold Change in Reproductive Number") + 
  theme(legend.position = c(0.75, 0.9), legend.background = element_blank())
p4

p5 <- ggplot(top_muts, aes(x = num_emergences, y = R_eff, label = mutation)) + 
  geom_jitter(width = 0.01, aes()) + 
  geom_text(data = subset(top_muts, rank < 10), size = 4, color = "black", position = position_jitter(w=0.05,h=0.05)) + 
  theme_bw() + 
  geom_smooth(method = "lm") + 
  labs( x = "Number of Emergences", y = "Fold Change in Reproductive Number") + 
  scale_x_log10() + 
  theme(legend.position = c(0.85, 0.85), legend.background = element_blank())
p5

m5 <- lm(top_muts$R_eff ~ top_muts$num_emergences)

p6 <- ggplot(top_muts, aes(x = num_emergences)) + 
  geom_histogram(bins = 10) + 
  theme_bw() + 
  scale_x_log10() + 
  labs(x = "Number of Emergenges")
p6

plot_grid(p6, p5, p4, labels = c("A", "B", "C"), nrow = 1)
ggsave("~/Dropbox/COVID/pyro-cov/paper/mutation_summaries.jpg", height = 5, width = 12)

# print specific effects for individual mutations

D614G <- top_muts %>% filter(mutation == "S:D614G")

