# Load library
library(readr)
library(dplyr)
library(ggplot2)

# Baca data
data <- read_csv("C:/Data D/BINUS FILES/AI/shop_data.csv")

# Lihat struktur data
head(data)
str(data)
summary(data)
dim(data)

# Tangani missing values
summary(is.na(data))
data <- data %>% drop_na()

# Hapus duplikat
data <- distinct(data)

# Filter data
filtered_data <- data %>% 
  filter(Selling_Price > 100)

# Grup dan agregasikan data
grouped_data <- data %>% 
  group_by(Delivery_Time) %>% 
  summarise(mean = mean(Final_Price))

# Visualisasikan data
ggplot(data, aes(x = Selling_Price, y = Final_Price)) + 
  geom_point() + 
  labs(title = "Hubungan Selling Price dan Final Price", 
       x = "Selling Price", 
       y = "Final Price")

# Statistik deskriptif
summary_statistik <- summary(data)
print(summary_statistik)

# Kuantil
quantile_statistik <- sapply(data, function(x) quantile(x, probs = c(0, 0.25, 0.5, 0.75, 1)))
print(quantile_statistik)

# Simpan statistik ke data frame
df <- data.frame(
  Initial_Price = data$initial_price,
  Selling_Price = data$selling_price,
  Delivery_Time = data$delivery_time,
  Final_Price = data$final_price
)

# Statistik deskriptif
summary_df <- summary(df)
print(summary_df)

# Kuantil
quantile_df <- sapply(df, function(x) quantile(x, probs = c(0, 0.25, 0.5, 0.75, 1)))
print(quantile_df)