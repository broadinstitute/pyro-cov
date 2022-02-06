### Create additional figures for pyro-cov paper
# April 13 2021
# lemieux@broadinstitute.org

library(tidyverse)
library(cowplot)
library(gt)
library(ggpubr)

top_strains <- read_tsv("~/Dropbox/COVID/pyro-cov/paper/strains.tsv")#[,1:3]
names(top_strains) <- c("rank" ,"strain", "Mean/SD","LogP","R_eff", "lower_ci","upper_ci","Cases_per_day", "Cases_total", "birthday", "mutations")
top_strains <- top_strains %>% 
  mutate(log_samples = log10(Cases_per_day)) %>% 
  mutate(VOC = ifelse(strain %in% c("P.1", "B.1.351", "B.1.617.2", "B.1.1.7"), "CDC VOC", "Non-VOC"))
top_strains$VOC[top_strains$strain %in% c("P.2", "B.1.427", "B.1.429", "B.1.525", "B.1.617.1", "B.1.617.3", "P.2","B.1.526")] <- "CDC VUI"
p1 <- ggplot(top_strains, aes(x = Cases_per_day, y = R_eff, label = strain, color = VOC)) + geom_text(size = 3) + 
  #geom_errorbarh(aes(xmin = lower_ci, xmax = upper_ci)) +
  theme_bw() + 
  #geom_smooth(method = "lm") + 
  labs(y = "Fold Increase in Reproductive Number", x = "Log10(Cases per Day)") + 
  scale_x_log10() + 
  theme(legend.position = c(0.2, 0.8))
p2 <- ggplot(top_strains, aes(x = R_eff)) + geom_histogram(binwidth = 0.05) + theme_bw() + 
  labs(x = "Fold Increase in Reproductive Number", y = "Count")

p3 <- ggplot(top_strains, aes(y = R_eff, x = birthday, label = strain, color = VOC)) +
  geom_text(size = 3) + theme_bw() + 
  labs( x= "Date of Lineage Emergence", y = "Fold Increase in Reproductive Number") + 
  #geom_smooth(aes(group = 1),method = "lm") + 
  theme(legend.position = c(0.2, 0.8))

p3
m1 <- lm(data = top_strains, Cases_per_day ~ R_eff)
summary(m1)
m2 <- lm(data = top_strains, R_eff ~ birthday)
summary(m2)

plot_grid(p2, p1,p3, nrow = 1, labels = c("A", "B", "C"))
ggsave("~/Dropbox/COVID/pyro-cov/paper/spectrum_transmissibility.jpg", height = 4, width = 12)

# construct table 2

top_muts <- read_tsv("~/Dropbox/COVID/pyro-cov/paper/mutations.tsv")
top_muts <- top_muts %>% mutate(num_emergences = 
                                  sapply(`emerged in lineages`, function(x) length(str_split(x, ",")[[1]]))) %>% 
  #mutate(CI = paste(round(`R / R_A 95% ci lower`, 2), round(`R / R_A 95% ci upper`, 2), sep="-")) %>% 
  mutate(R_eff = round(`R / R_A`, 2)) %>% 
  mutate(`mean/stddev` = round(`mean/stddev`, 2)) %>% 
  #mutate(`Fold Increase in R (CI)` = paste(`R_eff`, " (", `CI`,")", sep="")) %>% 
  mutate(ORF = sapply(mutation, function(x) strsplit(x, ":")[[1]][1])) %>% 
  mutate(Mutation = sapply(mutation, function(x) strsplit(x, ":")[[1]][2]))

muts_table <- top_muts[,c("rank", "ORF", "Mutation","log10(P(ΔR > 1))", "R / R_A",  "num_emergences", "emerged in lineages")]
names(muts_table) <- c("Rank","Gene", "Mutation", "LogP", "Fold Increase in Transmissibility", "Number of Lineages", "Emerged In")
t2 <- muts_table[1:20,] %>% gt() %>% 
  cols_align(align = "center") %>% tab_style(style = cell_text(weight = "bold"), 
                                             locations = cells_column_labels(columns = everything())
  ) %>% 
  fmt_number(columns = `Fold Increase in Transmissibility`, decimals = 2) %>%
  cols_hide(c(LogP, `Emerged In`))
t2

muts_to_include = 9

t3 <- muts_table %>% 
  filter(Gene == "N") %>%
  slice_head(n = muts_to_include)

  gt() %>% 
  cols_align(align = "center") %>% tab_style(style = cell_text(weight = "bold"), 
                                             locations = cells_column_labels(columns = everything())
#  ) %>% 
#  fmt_number(columns = LogP, decimals = 0) %>%
#  cols_hide(c(LogP))
t3

t4 <- muts_table %>% 
  filter(Gene == "ORF1b") %>%
  slice_head(n = muts_to_include)

#%>%
#  gt() %>% 
#  cols_align(align = "center") %>% tab_style(style = cell_text(weight = "bold"), 
#                                             locations = cells_column_labels(columns = everything())
#  ) %>% 
#  fmt_number(columns = LogP, decimals = 0) %>%
#  cols_hide(c(LogP))
t4

t5 <- muts_table %>% 
  filter(Gene == "S") %>%
  slice_head(n = muts_to_include) 


tcomb <- rbind(t3,t4,t5) 
tcomb$Gene <- gsub("N", "Nucleocapsid", tcomb$Gene)
tcomb$Gene <- gsub("S", "Spike", tcomb$Gene)
tcomb$Annotation = "             "

# 

gtcomb <- tcomb %>%
  gt(groupname_col = "Gene", rowname_col = "Annotation") %>% 
  cols_align(align = "center") %>% tab_style(style = cell_text(weight = "bolder"), 
                                             locations = cells_column_labels(columns = everything())
  ) %>% 
  tab_style(style = cell_text(weight="normal"), 
            locations = cells_row_groups()) %>%
  fmt_number(columns = LogP, decimals = 0) %>%
  cols_hide(c(LogP, `Emerged In`)) %>% 
  cols_align("left") %>% 
  fmt_number(columns = `Fold Increase in Transmissibility`, decimals = 2) %>% 
  tab_stubhead(label = "Open Reading Frame (ORF)          ") %>% 
  tab_style(style = cell_text(weight = "bold"), 
            locations = cells_stubhead())
gtcomb


# look at mutation occurrence summaries


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
strains_table <- top_strains[,c("rank", "strain","LogP", "R_eff", "lower_ci", "upper_ci", "Cases_per_day")]
strains_table <- strains_table %>% mutate(CI = paste(round(lower_ci, 2), round(upper_ci, 2), sep="-")) %>% 
  mutate(R_eff = round(R_eff, 2)) %>% 
  mutate(Cases_per_day = round(Cases_per_day, 0)) %>% 
  mutate(`Fold Increase in R (CI)` = paste(`R_eff`, " (", `CI`,")", sep=""))
strains_table <- strains_table[,c("rank", "strain","LogP", "Fold Increase in R (CI)", "Cases_per_day")]
names(strains_table) <- c("Rank", "Pango Lineage","LogP", "Fold Increase in R (CI)", "Cases per day (July 6 2021)")
t1 <- strains_table[1:10,] %>% gt() %>% 
  cols_align(align = "center") %>% tab_style(style = cell_text(weight = "bold"), 
                                             locations = cells_column_labels(columns = everything())
                                             ) %>% 
  fmt_number(columns = c(`Cases per day (July 6 2021)`, LogP), decimals = 0) %>% 
  cols_hide(c(LogP))
t1

  # plot 

top_muts <- top_muts %>% 
  mutate(gene = sapply(top_muts$mutation, function(x) str_split(x, ":")[[1]][1])) %>% 
  mutate(Effect = ifelse(top_muts$`R / R_A` > 1, "Increase Transmissibility", "Decrease Transmissibility")) %>%
  mutate(Rank = ifelse(top_muts$rank < 101, "Top 100", "Bottom 2132"))


p4 <- ggplot(data = subset(top_muts, R_eff > 0 & abs(`mean/stddev`) > quantile(abs(`mean/stddev`), 0.9)), aes(x = ORF, y = R_eff)) + 
  geom_violin(draw_quantiles = c(0.5), adjust = 2) + 
  geom_jitter(aes(color = Effect), width = 0.3, height = 0) + 
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 90)) + 
  labs( x = "Gene", y = "Fold Change in Reproductive Number") + 
  theme(legend.position = c(0.75, 0.9), legend.background = element_blank())
p4

p5 <- ggplot(top_muts, aes(x = num_emergences, y = R_eff, label = Mutation)) + 
  geom_jitter(width = 0.01, alpha = 0.2) + 
  geom_text(data = subset(top_muts, rank < 11), aes(color=ORF), size = 4, key_glyph =  draw_key_rect,
            position = position_jitter(w=0.05,h=0.05)) + 
  theme_bw() + 
  geom_smooth(method = "lm") + 
  labs( x = "Number of Emergences", y = "Fold Change in Reproductive Number") + 
  scale_x_log10() + 
  theme(legend.position = c(0.85, 0.65), legend.background = element_blank(), legend.text = element_text(size = 8))
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

muts_table <- muts_table %>% 
  mutate(Position = as.numeric(substr(Mutation, 2, nchar(Mutation) - 1)), 
         Effect = ifelse(`Fold Increase in Transmissibility` > 1, "Increase", "Decrease"))

p7 <- ggplot(muts_table, aes(x = Position, y = `Fold Increase in Transmissibility`, color = Effect)) + geom_point() + facet_grid(. ~ Gene) + 
  ylim(c(0.95, 1.05))

p7
p8 <- ggplot(muts_table %>% filter(Gene == "N"), aes(x = Position, y = `Fold Increase in Transmissibility`)) + geom_point()
p8

p9 <- ggplot(muts_table, aes(x = Position, y = `Fold Increase in Transmissibility`, color = Effect)) + geom_point() + facet_grid(Gene ~ .) + 
  ylim(c(0.95, 1.05)) + labs(y = "Fold Change in Transmissibility")

p9
ggsave("~/Dropbox/MGH/Grants/AViDD_U19/PyR0_all.pdf", height = 10, width = 10)


p10 <- ggplot(muts_table %>% filter(Gene == "S"), aes(x = Position, y = `Fold Increase in Transmissibility`, color = Effect)) + geom_point()
p10

p11 <- ggplot(muts_table %>% filter(Gene == "ORF3a"), aes(x = Position, y = `Fold Increase in Transmissibility`, color = Effect)) + geom_point()
p11

ggsave("~/Dropbox/MGH/Grants/AViDD_U19/PyR0_ORF3a.pdf", height = 3, width = 6)


p12 <- ggplot(muts_table %>% filter(Gene == "ORF8"), aes(x = Position, y = `Fold Increase in Transmissibility`, color = Effect)) + geom_point()
p12


p13 <- ggplot(muts_table, aes(y = `Fold Increase in Transmissibility`, x = Gene)) +
  geom_violin()
p13

p14 <- ggplot(muts_table %>% filter(Gene == "N"), aes(x = Position, y = `Fold Increase in Transmissibility`, color = Effect)) + 
  geom_point() + 
  theme_bw() + labs(title = expression(PyR["0"]*" Mutation-Level Estimates of Transmissibility: Nucleocapsid")) + 
  theme(legend.position = c(0.2, 0.7), legend.background = element_blank(), legend.key = element_blank())
p14
ggsave("~/Dropbox/MGH/Grants/AViDD_U19/PyR0_nucleocapsid.pdf", height = 3, width = 6)

p15 <- ggplot(top_muts %>% filter(`R_eff` < 1), aes(x = `R_eff`, y = abs(`mean/stddev`), label = mutation))+ 
  geom_label() + theme_bw() + xlim(c(0.8, 1)) +
  labs(x = "Fold Change in Growth Rate", y = "Significance", title = expression(PyR["0"]*" Mutations Predicted to Decrease Growth Rate"))
p15
ggsave("~/Dropbox/MGH/Grants/AViDD_U19/PyR0_volcano_left.pdf", height = 5, width = 5)


# compare multinomial fits vs pyr0

multinom_fit <- read_csv("~/Dropbox/COVID/learn_pyro/multin_R_fit_01_21_2022.csv")
names(multinom_fit)[1] <- "strain"
merged_fit <- left_join(top_strains, multinom_fit, by = "strain")
merged_fit <- merged_fit %>% mutate(expWeek = exp(Week))

ggplot(merged_fit, aes(x = `R_eff`/1.6, y = expWeek, label = strain)) + 
  geom_text() + theme_bw() + 
  geom_smooth(method = lm) + 
  labs(y = "Growth Rate Relative to B.1.1.7 (Multinomial LR)", x = "Growth Rate Relative to B.1.1.7 (PyR0)")

ggsave("~/Dropbox/COVID/pyro-cov/paper/multinomial_LR_vs_pyr0.jpg", height = 5, width = 5)
cor(merged_fit$R_eff/1.6, merged_fit$expWeek, use = "complete.obs")

