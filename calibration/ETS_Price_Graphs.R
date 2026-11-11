# ETS Price Graphs

data <- Final_dataset

#Check data types
str(data)

# Convert all columns except the first (Trading Date) to numeric
data[, -1] <- lapply(data[, -1], as.numeric)

# Convert first column to Date
data[[1]] <- as.Date(data[[1]])

# Check data types
head(data)

library(dplyr)
install.packages("tidyr")
library(tidyr)
library(ggplot2)
install.packages("viridis")
library(viridis)
install.packages("showtext")
library(showtext)
install.packages("sysfonts")
library(sysfonts)

# Fira Sans
font_add_google("Fira Sans", "fira")
showtext_auto()

# Extract maturity year from column names
maturity_years <- as.numeric(sub(".*-", "", names(data)[-1]))

#In case you want to take only a subset of the full time frame:
# # Keep only maturities expiring in 2021 or later
# keep_cols <- c(
#   TRUE,  # Keep Date column
#   maturity_years >= 2021
# )
# 
# data_plot <- data[, keep_cols]
# 
# # Restrict observations to 2021 onwards
# data_plot <- subset(
#   data_plot,
#   Date >= as.Date("2021-01-01")
# )

# Comment out if want to filter for a subset of years
data_plot <- data

# Colours
cols <- viridis(ncol(data_plot) - 1)

# Moloch-style appearance
par(
  family = "fira",
  bg = "#FAFAFA",
  fg = "black",
  bty = "l"
)

# Plot
matplot(
  data_plot$Date,
  data_plot[, -1],
  type = "l",
  lty = 1,
  lwd = 1,
  col = cols,
  xaxt = "n",
  xlab = "Date",
  ylab = "ETS Futures Price",
  main = "ETS Futures Prices by Maturity"
)

# Monthly ticks
axis.Date(
  side = 1,
  at = seq(
    min(data_plot$Date), #as.Date("2010-01-01"), # Can adjust this 
    max(data_plot$Date),
    by = "3 months"
  ),
  format = "%Y-%m",
  las = 2,
  cex.axis = 0.6
)

# Legend
legend(
  "topleft",
  legend = names(data_plot)[-1],
  col = cols,
  lty = 1,
  cex = 0.7,
  bty = "n"
)

# Second plot
# All Maturity Horizons in 1 plot
maturity_dates <- as.Date(
  paste0(
    sub("Dec-", "", names(data_plot)[-1]),
    "-12-31"
  )
)

# Initialise an empty matrix for the full horizon
max_horizon <- 730

horizon_matrix <- matrix(
  NA,
  nrow = nrow(data_plot),
  ncol = max_horizon
)

# Fill the matrix
horizon_matrix <- matrix(
  NA,
  nrow = nrow(data_plot),
  ncol = max_horizon
)

for (i in seq_len(nrow(data_plot))) {

  trade_date <- data_plot$Date[i]

  available <- !is.na(data_plot[i, -1])

  if (any(available)) {

    ttm_days <- as.integer(
      maturity_dates[available] - trade_date
    )

    prices <- as.numeric(
      data_plot[i, -1][available]
    )

    valid <- ttm_days >= 1 &
      ttm_days <= max_horizon

    for (j in which(valid)) {

      horizon_matrix[
        i,
        ttm_days[j]
      ] <- prices[j]

    }

  }

}

# To check how many non-NA's are in th resulting matrix
coverage <- colSums(!is.na(horizon_matrix))

# Base plot (empty points, just establishes axes/theme)
matplot(
  data_plot$Date,
  horizon_matrix,
  type = "n",
  xaxt = "n",
  xlab = "Trading Date",
  ylab = "Price (€/tCO₂)",
  main = "ETS Prices by Exact Time-to-Maturity"
)

# Small tick every month
axis.Date(
  1,
  at = seq(
    min(data_plot$Date),
    max(data_plot$Date),
    by = "month"
  ),
  labels = FALSE,
  tcl = -0.2
)

# Label every quarter
axis.Date(
  1,
  at = seq(
    min(data_plot$Date),
    max(data_plot$Date),
    by = "3 months"
  ),
  format = "%Y-%m",
  las = 2,
  cex.axis = 0.6
)


# Add all horizons
for(h in 1:max_horizon){

  idx <- !is.na(horizon_matrix[, h])

  points(
    data_plot$Date[idx],
    horizon_matrix[idx, h],
    pch = 16,
    cex = 0.8,
    col = viridis(max_horizon)[h]
  )

}

cols <- adjustcolor(
  viridis(max_horizon),
  alpha.f = 0.30
)

for(h in 1:max_horizon){

  # Selecting all trading dates where that horizon exists
  idx <- !is.na(horizon_matrix[, h])

  if(sum(idx) >= 2){

    # Joining corresponding horizons with lines
    lines(
      data_plot$Date[idx],
      horizon_matrix[idx, h],
      col = cols[h],
      lwd = 1
    )

  }

}

# Adjust the reference horizons in accordance to the max_horizon variable
horiz_ref <- c(1, 15, 30, 45, 60, 75, 90, 180, 365, 730)

legend(
  "topleft",
  legend = paste(horiz_ref, "days"),
  col = viridis(max_horizon)[horiz_ref],
  pch = 16,
  pt.cex = 1.2,
  bty = "n",
  title = "Time to Maturity"
)
 