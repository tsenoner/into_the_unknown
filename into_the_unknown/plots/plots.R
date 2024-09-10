library(httr)
library(jsonlite)
library(openssl)
library(RColorBrewer)
library(data.table)
library(ggplot2)
library(dplyr)
library(ggforce)
library(scales)
class <- c("Phylum", "Class", "Order", "Family", "Genus", "Species")

get_hit_n <- function(api_response){
  return(fromJSON(content(api_response, "text"))$status$hits)
}

rbindlist(lapply(class, function(x){
  q = paste("assembly_span AND tax_rank(", x, ")", sep = "")
  res_all = GET("https://goat.genomehubs.org/", path = "api/v2/search",
            query = list(query = q, result = "taxon", taxonomy = "ncbi", includeEstimates = "true", fields = "assembly_span"))
  res_avl = GET("https://goat.genomehubs.org/", path = "api/v2/search",
                query = list(query = q, result = "taxon", taxonomy = "ncbi", includeEstimates = "true", fields = "assembly_span",
                             excludeMissing = c("assembly_span"), excludeAncestral = c("assembly_span")))
  return(data.table("class" = x, "available" = get_hit_n(res_avl), "total" = get_hit_n(res_all)))
})) -> dt

dt[, percentage := available/total]
dt$class <- factor(dt$class, levels = rev(dt$class))

formatter <- label_number(scale_cut = cut_short_scale(), accuracy = 0.01)

dt[, label := paste(
  scales::percent(percentage, accuracy = 0.01), 
  "\n(",
  lapply(available, function(x) {formatter(x)}), "/", 
  lapply(total, function(x) {formatter(x)}), 
  ")", 
  sep = ""
)]





ggplot(dt, aes(x = class, y = percentage)) +
  geom_bar(data = dt, aes(x = class, y = 1), stat = "identity", width = 1, fill = "grey60") +
  geom_bar(stat = "identity", width = 1, aes(fill = class)) +
  geom_bar(data = dt, aes(x = class, y = 1), stat = "identity", width = 1, fill = "grey80", color = "grey40", alpha = 0, lwd = 1) +
  scale_y_continuous(limits = c(0, 1)) +
  geom_text(data = dt, aes(x = class, y = 0.35, label = label),size = 8, color = "white") +
  theme_void() +
  guides(fill = guide_legend(nrow = 6, reverse = TRUE))+
  labs(title = "Taxa with assemblies out of all Eukaryotic taxa in INSDC",
       x = NULL,
       y = NULL,
       fill = NULL) +
  theme(legend.position = c(0.07, 0.5),
        legend.key.width = unit(1.5, "line"),
        legend.key.spacing.y = unit(150, "pt"),
        plot.title = element_text(hjust = 0.5, size = 32),
        legend.text = element_text(size = 24)) + 
  ylim(-0.1,1) +
  scale_fill_brewer(palette = "PuBuGn")+
  coord_flip()
#export 1500x1500




###histogram for the count of assemblies 


res_all = GET("https://goat.genomehubs.org/", path = "api/v2/search",
              query = list(query = "assembly_date >= 2006-01-01 AND assembly_date < 2025-01-01 AND tax_rank(species) AND assembly_level=scaffold,contig,chromosome,complete genome", 
                           result = "taxon", taxonomy = "ncbi", includeEstimates = "false", fields="assembly_date, assembly_level", 
                           excludeMissing = "assembly_date", excludeMissing = "assembly_span", excludeAncestral = c("assembly_span"), size = 10000000))
dt <- data.table(fromJSON(content(res_all, "text"))$results)

dt[, assembly_level := dt$result.fields.assembly_level.value]
dt[, assembly_year := year(as.Date(result.fields.assembly_date.value, format = "%Y-%m-%d"))]


bin_counts <- dt[, .N, by = .(assembly_year, assembly_level)][, prop := N / sum(N), by = assembly_year]
bin_counts$assembly_level <- factor(bin_counts$assembly_level, levels = c("Complete Genome", "Contig", "Chromosome", "Scaffold"))


total_elements <- sum(bin_counts$N)
current_date <- format(Sys.Date(), "%Y-%m-%d")

label_counts <- bin_counts[,sum(N), by=assembly_level]
label_counts <- label_counts[order(label_counts$assembly_level), ]
legend_labels <- paste(label_counts$assembly_level, "\nn = ", label_counts$V1, sep = "")



# Create the histogram by year
p <- ggplot(bin_counts, aes(x = assembly_year, fill = assembly_level, weight = N)) +
  geom_histogram(binwidth = 1, position = "stack", color = "black") +
  scale_fill_manual(name = "",  # Add total count to legend title
                    values = c("seagreen3", "turquoise3", "royalblue1", "midnightblue"),
                    labels = legend_labels) +
  labs(title = paste0("Assembly Dates by Year as of ", current_date),
       x = "Assembly Year",
       y = "Count") +
  theme_minimal() +
  theme(legend.key.spacing.y = unit(20, "pt")) +
  scale_x_continuous(breaks = seq(floor(min(bin_counts$assembly_year)), ceiling(max(bin_counts$assembly_year)), by = 2))


# Add the color legend
p + guides(fill = guide_legend(override.aes = list(color = NULL)))  
