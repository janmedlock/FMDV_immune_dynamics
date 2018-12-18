# FMDV, Birth-pulse data. Figures
setwd("~/GitHub/FMDV-birth-pulse")
library(ggplot2)
library(RColorBrewer)
library(png)

#############################################
#############################################
# Figure 1
############################################
############################################

# Figure 1.1 birth data
############################################
birthdata <- readRDS("birth_pulse/birthdata.rds")
newdf <- data.frame(
    order = rep(seq(1, 12, 1), 3),
    month = rep(c(7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6), 3), 
    numbirths = birthdata$numsurvbirths[7:42], 
    birthingseason = c(rep("2013-2014", 12), rep("2014-2015", 12),
        rep("2015-2016", 12)) )
month_table <- table(newdf$month)
newdf$month2 <- factor(newdf$month, 
    levels = names(month_table)[order(newdf$month[1:12])])

newdf <- data.frame(
    order = rep(seq(1, 12, 1), 2),
    month = rep(c(7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6), 2), 
    numbirths = birthdata$numbirths[7:30], 
    birthingseason = c(rep("2013-2014", 12), rep("2014-2015", 12) ))
month_table <- table(newdf$month)
newdf$month2 <- factor(newdf$month, 
                       levels = names(month_table)[order(newdf$month[1:12])])

blues <- brewer.pal(8, "Blues")

# Data plot
p1.1<- ggplot(newdf, aes(x = month2, y = numbirths, fill = birthingseason)) + 
    geom_bar(stat="identity", color = "black") + 
    xlab("Month") + ylab("Number of births") + 
    theme_bw() +
    scale_fill_manual(values = c(blues[8], blues[7]), name = "birthing season") + 
    scale_x_discrete(labels = c("7" = "Jul", "8" = "", "9" = "",
        "10" = "Oct", "11" = "", "12" = "", "1" = "Jan", "2" = "", 
        "3" = "", "4" = "Apr", "5" = "", "6" = "")) +
    theme(axis.line.x = element_line(colour= "black"),
        axis.line.y = element_line(colour= "black"),
        axis.title.x= element_text(size = 15, vjust=-0.15), 
        axis.title.y = element_text(size = 15, vjust= 0.8),
        axis.text.x = element_text(size = 15, vjust=-0.05),
        axis.text.y = element_text(size = 14),
        legend.text = element_text(size = 12),
        legend.title = element_blank(),
        legend.position=c(0.2, 0.9),
        panel.border = element_blank(), 
        axis.line = element_line(colour= "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

# Figure 1.2 maternal immunity data
############################################
m <- read.csv("~/Documents/postdoc_buffology/FMDV paper/maternal_immunity.csv")
mdf <- m[m$remove == "n", c(1:5)]
mdf <- mdf[!is.na(mdf$age), ]
mdf$sid <- as.factor(paste(as.character(mdf$sat), 
    as.character(mdf$id), sep = "_"))

p1.2<- ggplot(mdf, aes(x = age, y = titer, shape = sat, colour = sat, group = sid)) + 
    geom_line() +
    geom_point() + xlim(0, 0.8) + ylim(1.2, 2.3) + 
    scale_color_manual(values = c(reds[8], reds[6], reds[4])) +  
    xlab("Age (years)") + ylab("Maternal antibody titer") + 
    theme_bw() +
    theme(axis.line.x = element_line(colour= "black"),
          axis.line.y = element_line(colour= "black"),
          axis.title.x= element_text(size = 15, vjust=-0.15), 
          axis.title.y = element_text(size = 15, vjust= 0.8),
          axis.text.x = element_text(size = 15, vjust=-0.05),
          axis.text.y = element_text(size = 14),
          legend.text = element_text(size = 12),
          legend.title = element_blank(),
          legend.key = element_blank(),
          legend.position=c(0.9, 0.9),
          panel.border = element_blank(), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.line = element_line(colour= "black")) + 
    geom_hline(yintercept = 1.7, colour = "black", linetype = 2)


# Figure 1.3 Buffalo picture
############################################
img1.3 <- readPNG("~/Documents/postdoc_buffology/FMDV paper/fig1.3_long.png")
# convert raster to image
grob <- rasterGrob(img1.3)


# Figure 1.4 - 1.6
############################################
source("birth_pulse/birth_pulse_functions.R")
sigma<-seq(0,1, 5)
mu= 0.443
t0 = 0.0416667  # mid January

plotdf<- data.frame(tval = rep(c(round(seq(0, 2, 0.001) + 0.0416666, digits = 4)), 2), 
    hazard = NA, 
    cv = c(rep(0.52, 2001), rep(0.995, 2001)), 
    birthingseason = c(rep("2013-2014", 2001), 
        rep("2014-2015", 2001)) )
for (i in 1:length(plotdf[ ,1])) {
    plotdf$hazard[i] <- get_hazard_birth(
        cv = plotdf$cv[i], time = plotdf$tval[i])
}
plotdf$month <- 1 + (plotdf$tval - 0.0417) * 12
df <- plotdf[plotdf$month > 6.9 & plotdf$month < 18.5, ]  # january = 0
df$year <- df$month / 12
p1.4 <- ggplot(df, aes(x = month, y = hazard, colour = birthingseason)) + 
    geom_line() + 
    scale_color_manual(values = c(blues[8], blues[7]), guide = FALSE) + 
    xlab("Month") + ylab("Birth hazard") + 
    theme_bw()  +
    scale_x_continuous(breaks = seq(7, 18, 1), 
        labels = c("Jul","", "", "Oct","", "","Jan","", "", 
            "Apr", "", "")) +
    theme(axis.line.x = element_line(colour= "black"),
          axis.line.y = element_line(colour= "black"),
          axis.title.x= element_text(size = 15, vjust=-0.15), 
          axis.title.y = element_text(size = 15, vjust= 0.8),
          axis.text.x = element_text(size = 15, vjust=-0.05),
          axis.text.y = element_text(size = 14),
          legend.position=c(0.9, 0.9),
          legend.title = element_blank(),
          legend.key = element_blank(),
          legend.text = element_text(size = 14),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(), axis.line = 
            element_line(colour= "black")  ) 

# model parameters in Jan's figure
#jan <- read.csv("birth_pulse/recruitment_rate.csv")
#jan <- jan[!is.na(jan$birth), ]
#jan$month <- jan$X * 12
#p1.4 <- ggplot(jan, aes(x = month, y = birth)) + 
#    geom_line(colour = blues[8]) + 
#    xlab("Month") + ylab("Birth") + 
#    theme_bw() + xlim(1, 12.1) +
#    scale_x_continuous(breaks = seq(1, 1.04, 1/12)) +
#    theme(axis.line.x = element_line(colour= "black"),
#          axis.line.y = element_line(colour= "black"),
#          axis.title.x= element_text(size = 15, vjust=-0.15), 
#          axis.title.y = element_text(size = 15, vjust= 0.8),
#          axis.text.x = element_text(size = 15, vjust=-0.05),
#          axis.text.y = element_text(size = 15),
#          legend.position=c(0.9, 0.9),
#          legend.title = element_blank(),
#          legend.key = element_blank(),
#          legend.text = element_text(size = 14),
#          panel.border = element_blank(), axis.line = 
#              element_line(colour= "black")  ) 
#p1.4

# Figure 1.5 Birth probability
############################################
# gamma distribution with transparency or just outline... 
# represent wanning maternal immunity
reds <- brewer.pal(8, "Reds")
x <- seq(0, 1, 0.00001)

df = data.frame(
    month = rep(x * 18, 3),
    density =c(dgamma(x, shape = 1.01, scale = 0.35), 
               dgamma(x, shape = 1.71, scale = 0.39), 
               dgamma(x, shape = 1.21, scale = 0.37)), 
    sat = c(rep("SAT-1", length(x)), rep("SAT-2", length(x)), 
            rep("SAT-3", length(x))) )
df$year = df$month / 12

p1.5 <- ggplot(df, aes(x = year, y = density, color = sat)) + 
    geom_line() + 
    xlab("Duration of maternal immunity") + ylab("Density") + 
    theme_bw() + xlim(0, 1.5) + 
    scale_colour_manual(values = c(reds[8], reds[6], reds[4]), guide = FALSE) + 
    theme(axis.line.x = element_line(colour= "black"),
        axis.line.y = element_line(colour= "black"),
        axis.title.x= element_text(size = 15, vjust=-0.15), 
        axis.title.y = element_text(size = 15, vjust= 0.8),
        axis.text.x = element_text(size = 15, vjust=-0.05),
        axis.text.y = element_text(size = 14),
        legend.position=c(0.9, 0.9),
        legend.title = element_blank(),
        legend.key = element_blank(),
        legend.text = element_text(size = 14),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour= "black")  ) 

jan <- read.csv("birth_pulse/recruitment_rate.csv")
jan <- jan[!is.na(jan$birth), ]
jan$month <- jan$X * 12
p1.6 <- ggplot(jan, aes(x = month, y = susceptible)) + 
    geom_line(colour = blues[8]) + 
    xlab("Month") + ylab("Birth") + 
    theme_bw() + xlim(1, 12.1) +
    scale_x_continuous(breaks = seq(1, 1.04, 1/12)) +
    theme(axis.line.x = element_line(colour= "black"),
          axis.line.y = element_line(colour= "black"),
          axis.title.x= element_text(size = 15, vjust=-0.15), 
          axis.title.y = element_text(size = 15, vjust= 0.8),
          axis.text.x = element_text(size = 15, vjust=-0.05),
          axis.text.y = element_text(size = 14),
          legend.position=c(0.9, 0.9),
          legend.title = element_blank(),
          legend.key = element_blank(),
          legend.text = element_text(size = 14),
          panel.border = element_blank(), axis.line = 
              element_line(colour= "black")  ) 
p1.6


multiplot(p1.1, p1.4, p1.2, p1.5, cols=2)
# png size at 

#############################################
#############################################
# Figure 2
############################################
############################################
# Acute transmission parameters
data.dir <- "~/Documents/postdoc_buffology/FMDV paper/ParameterEstimation/BuffaloTransmissionExperiments_FinalResults/"
s1 <- read.table(paste(data.dir, "SAT1_SEIRModel_MCMCSamples.txt", sep = ""),
    header = FALSE)
s2 <- read.table(paste(data.dir, "SAT2_SEIRModel_MCMCSamples.txt", sep = ""), 
    header = FALSE)
s3 <- read.table(paste(data.dir, "SAT3_SEIRModel_MCMCSamples.txt", sep = ""), 
    header = FALSE)
colnames <- c(rep("timeofinfection", 4), rep("latent.period", 4), 
    rep("infectious.period", 8), "shape.latent", 
	"mean.latent", "shape.infectious", "mean.infectious", "transmission")
colnames(s3) <- colnames(s2) <- colnames(s1) <- colnames
colnames(s2) <- colnames
# CHECK ME
s1$R <- s1$transmission * s1$mean.infectious
s2$R <- s2$transmission * s2$mean.infectious
s3$R <- s3$transmission * s3$mean.infectious



length(s1[,1]) #20000
# something doesn't match here - should have 2,000,000 iteractions after a 
# 2,000,000 burn in; thinned by taking every 200th sample
s1 <- tail(s1, 10000)
s2 <- tail(s2, 10000)
s3 <- tail(s3, 10000)
  
purples <- brewer.pal(8, "Purples")
reds <- brewer.pal(8, "Reds")
blues <- brewer.pal(8, "Blues")

fill <- c(blues[7], reds[6], purples[6])
colour <- c(blues[8],  reds[8],  purples[8])

# Figure 2.1 - mean latent period
##############################################
df <- data.frame(
	mean = c(s1$mean.latent, s2$mean.latent, s3$mean.latent), 
	sat = c(rep("SAT-1", length(s1$mean.latent)), 
        rep("SAT-2", length(s2$mean.latent)), 
		rep("SAT-3", length(s3$mean.latent)) ), 
	median = c(rep(median(s1$mean.latent), length(s1$mean.latent)), 
	    rep(median(s2$mean.latent), length(s2$mean.latent)), 
		rep(median(s3$mean.latent), length(s3$mean.latent)) ), 
	upper = c( rep(quantile(s1$mean.latent, 0.75), length(s1$mean.latent)), 
	    rep(quantile(s2$mean.latent, 0.75), length(s2$mean.latent)), 
	    rep(quantile(s3$mean.latent, 0.75), length(s3$mean.latent))  ), 
	lower = c( rep(quantile(s1$mean.latent, 0.25), length(s1$mean.latent)), 
	    rep(quantile(s2$mean.latent, 0.25), length(s2$mean.latent)), 
		rep(quantile(s3$mean.latent, 0.25), length(s3$mean.latent))  )	)

p2.1 <- ggplot(df, aes(x = sat, y = mean, colour = sat, fill = sat)) + 
    geom_violin(adjust = 2) + 
    scale_fill_manual(values = fill, guide = F) + scale_colour_manual(values = colour, guide = F) + 
    xlab("") + ylab("Mean latent period") + 
    theme_bw() + 
    scale_y_continuous(limits = c(0, 11.9), breaks = c(0, 2, 4, 6, 8, 10, 12)) + 
    theme(axis.line.x = element_line(colour= "black"),
          axis.line.y = element_line(colour= "black"),
          axis.title.x= element_text(size = 12, vjust=-0.15), 
          axis.title.y = element_text(size = 15, vjust= 0.8),
          axis.text.x = element_text(size = 12, vjust=-0.05),
          axis.text.y = element_text(size = 14),
          legend.position=c(0.9, 0.9),
          legend.title = element_blank(),
          legend.key = element_blank(),
          legend.text = element_text(size = 14),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(), axis.line = 
              element_line(colour= "black")  ) + 
    geom_point(data = df, aes(x = sat, y = median), colour = "black") +
    geom_linerange(data = df, aes(ymin = lower, ymax = upper))

# Figure 2.4 - latent period shape
##############################################
df <- data.frame(
    mean = c(s1$shape.latent, s2$shape.latent, s3$shape.latent), 
    sat = c(rep("SAT-1", length(s1$shape.latent)), 
            rep("SAT-2", length(s2$shape.latent)), 
            rep("SAT-3", length(s3$shape.latent)) ), 
    median = c(rep(median(s1$shape.latent), length(s1$shape.latent)), 
               rep(median(s2$shape.latent), length(s2$shape.latent)), 
               rep(median(s3$shape.latent), length(s3$shape.latent)) ), 
    upper = c( rep(quantile(s1$shape.latent, 0.75), length(s1$shape.latent)), 
               rep(quantile(s2$shape.latent, 0.75), length(s2$shape.latent)), 
               rep(quantile(s3$shape.latent, 0.75), length(s3$shape.latent))  ), 
    lower = c( rep(quantile(s1$shape.latent, 0.25), length(s1$shape.latent)), 
               rep(quantile(s2$shape.latent, 0.25), length(s2$shape.latent)), 
               rep(quantile(s3$shape.latent, 0.25), length(s3$shape.latent))  )	)

p2.4 <- ggplot(df, aes(x = sat, y = mean, colour = sat, fill = sat)) + 
    geom_violin(adjust = 2) + 
    scale_fill_manual(values = fill, guide = F) + scale_colour_manual(values = colour, guide = F) + 
    xlab("") + ylab("Latent period shape") + 
    theme_bw() + 
    scale_y_continuous(limits = c(0, 11.9), breaks = c(0, 2, 4, 6, 8, 10, 12)) + 
    theme(axis.line.x = element_line(colour= "black"),
          axis.line.y = element_line(colour= "black"),
          axis.title.x= element_text(size = 12, vjust=-0.15), 
          axis.title.y = element_text(size = 15, vjust= 0.8),
          axis.text.x = element_text(size = 12, vjust=-0.05),
          axis.text.y = element_text(size = 14),
          legend.position=c(0.9, 0.9),
          legend.title = element_blank(),
          legend.key = element_blank(),
          legend.text = element_text(size = 14),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(), axis.line = 
              element_line(colour= "black")  ) + 
    geom_point(data = df, aes(x = sat, y = median), colour = "black") +
    geom_linerange(data = df, aes(ymin = lower, ymax = upper))


# Figure 2.2 - mean infectious period
##############################################
df <- data.frame(
	mean = c(s1$mean.infectious, s2$mean.infectious, s3$mean.infectious), 
	sat = c(rep("SAT-1", length(s1$mean.infectious)), rep("SAT-2", length(s2$mean.infectious)), 
		rep("SAT-3", length(s3$mean.infectious)) ), 
	median = c(rep(median(s1$mean.infectious), length(s1$mean.infectious)), 
		rep(median(s2$mean.infectious), length(s2$mean.infectious)), 
		rep(median(s3$mean.infectious), length(s3$mean.infectious)) ), 
	upper = c( rep(quantile(s1$mean.infectious, 0.75), length(s1$mean.infectious)), 
		rep(quantile(s2$mean.infectious, 0.75), length(s2$mean.infectious)), 
		rep(quantile(s3$mean.infectious, 0.75), length(s3$mean.infectious))  ), 
	lower = c( rep(quantile(s1$mean.infectious, 0.25), length(s1$mean.infectious)), 
		rep(quantile(s2$mean.infectious, 0.25), length(s2$mean.infectious)), 
		rep(quantile(s3$mean.infectious, 0.25), length(s3$mean.infectious))  )	)

p2.2 <- ggplot(df, aes(x = sat, y = mean, colour = sat, fill = sat)) + 
    geom_violin(adjust = 2) + 
    scale_fill_manual(values = fill, guide = F) + scale_colour_manual(values = colour, guide = F) + 
    xlab("") + ylab("Mean infectious period") + 
    theme_bw() + 
    scale_y_continuous(limits = c(0, 11.9), breaks = c(0, 2, 4, 6, 8, 10, 12)) + 
    theme(axis.line.x = element_line(colour= "black"),
          axis.line.y = element_line(colour= "black"),
          axis.title.x= element_text(size = 12, vjust=-0.15), 
          axis.title.y = element_text(size = 15, vjust= 0.8),
          axis.text.x = element_text(size = 12, vjust=-0.05),
          axis.text.y = element_text(size = 14),
          legend.position=c(0.9, 0.9),
          legend.title = element_blank(),
          legend.key = element_blank(),
          legend.text = element_text(size = 14),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(), axis.line = 
              element_line(colour= "black")  ) + 
    geom_point(data = df, aes(x = sat, y = median), colour = "black") + 
    geom_linerange(data = df, aes(ymin = lower, ymax = upper))

# Figure 2.5 - shape infectious period
##############################################
df <- data.frame(
    mean = c(s1$shape.infectious, s2$shape.infectious, s3$shape.infectious), 
    sat = c(rep("SAT-1", length(s1$shape.infectious)), rep("SAT-2", length(s2$shape.infectious)), 
            rep("SAT-3", length(s3$shape.infectious)) ), 
    median = c(rep(median(s1$shape.infectious), length(s1$shape.infectious)), 
               rep(median(s2$shape.infectious), length(s2$shape.infectious)), 
               rep(median(s3$shape.infectious), length(s3$shape.infectious)) ), 
    upper = c( rep(quantile(s1$shape.infectious, 0.75), length(s1$shape.infectious)), 
               rep(quantile(s2$shape.infectious, 0.75), length(s2$shape.infectious)), 
               rep(quantile(s3$shape.infectious, 0.75), length(s3$shape.infectious))  ), 
    lower = c( rep(quantile(s1$shape.infectious, 0.25), length(s1$shape.infectious)), 
               rep(quantile(s2$shape.infectious, 0.25), length(s2$shape.infectious)), 
               rep(quantile(s3$shape.infectious, 0.25), length(s3$shape.infectious))  )	)

p2.5 <- ggplot(df, aes(x = sat, y = mean, colour = sat, fill = sat)) + 
    geom_violin(adjust = 2) + 
    scale_fill_manual(values = fill, guide = F) + scale_colour_manual(values = colour, guide = F) + 
    xlab("") + ylab("Infectious period shape") + 
    theme_bw() + 
    scale_y_continuous(limits = c(0, 11.9), 
        breaks = c(0, 2, 4, 6, 8, 10, 12)) + 
#    scale_y_continuous(limits = c(0, 15.9), 
#        breaks = c(0, 2, 4, 6, 8, 10, 12, 14, 16)) + 
    theme(axis.line.x = element_line(colour= "black"),
          axis.line.y = element_line(colour= "black"),
          axis.title.x= element_text(size = 12, vjust=-0.15), 
          axis.title.y = element_text(size = 15, vjust= 0.8),
          axis.text.x = element_text(size = 12, vjust=-0.05),
          axis.text.y = element_text(size = 14),
          legend.position=c(0.9, 0.9),
          legend.title = element_blank(),
          legend.key = element_blank(),
          legend.text = element_text(size = 14),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(), axis.line = 
              element_line(colour= "black")  ) + 
    geom_point(data = df, aes(x = sat, y = median), colour = "black") + 
    geom_linerange(data = df, aes(ymin = lower, ymax = upper))


# Figure 2.3 transmission rate
##############################################
df <- data.frame(
    mean = c(s1$transmission, s2$transmission, s3$transmission), 
    sat = c(rep("SAT-1", length(s1$transmission)), rep("SAT-2", length(s2$transmission)), 
            rep("SAT-3", length(s3$transmission)) ), 
    median = c(rep(median(s1$transmission), length(s1$transmission)), 
               rep(median(s2$transmission), length(s2$transmission)), 
               rep(median(s3$transmission), length(s3$transmission)) ), 
    upper = c( rep(quantile(s1$transmission, 0.75), length(s1$transmission)), 
               rep(quantile(s2$transmission, 0.75), length(s2$transmission)), 
               rep(quantile(s3$transmission, 0.75), length(s3$transmission))  ), 
    lower = c( rep(quantile(s1$transmission, 0.25), length(s1$transmission)), 
               rep(quantile(s2$transmission, 0.25), length(s2$transmission)), 
               rep(quantile(s3$transmission, 0.25), length(s3$transmission))  )	)


p2.3 <- ggplot(df, aes(x = sat, y = mean, colour = sat, fill = sat)) + 
    geom_violin(adjust = 2) + 
    scale_fill_manual(values = fill, guide = F) + scale_colour_manual(values = colour, guide = F) + 
    xlab("") + ylab("Transmission rate") + 
    theme_bw() + 
    scale_y_continuous(limits = c(0, 35), 
        breaks = c(0, 5, 10, 15, 20, 25, 30, 35)) + 
    theme(axis.line.x = element_line(colour= "black"),
          axis.line.y = element_line(colour= "black"),
          axis.title.x= element_text(size = 12, vjust=-0.15), 
          axis.title.y = element_text(size = 15, vjust= 0.8),
          axis.text.x = element_text(size = 12, vjust=-0.05),
          axis.text.y = element_text(size = 14),
          legend.position=c(0.9, 0.9),
          legend.title = element_blank(),
          legend.key = element_blank(),
          legend.text = element_text(size = 14),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(), axis.line = 
              element_line(colour= "black")  ) + 
    geom_point(data = df, aes(x = sat, y = median), colour = "black") + 
    geom_linerange(data = df, aes(ymin = lower, ymax = upper))    

# Figure 2.6 Ro
##############################################
df <- data.frame(
    mean = c(s1$R, s2$R, s3$R), 
    sat = c(rep("SAT-1", length(s1$R)), rep("SAT-2", length(s2$R)), 
            rep("SAT-3", length(s3$R)) ), 
    median = c(rep(median(s1$R), length(s1$R)), 
               rep(median(s2$R), length(s2$R)), 
               rep(median(s3$R), length(s3$R)) ), 
    upper = c( rep(quantile(s1$R, 0.75), length(s1$R)), 
               rep(quantile(s2$R, 0.75), length(s2$R)), 
               rep(quantile(s3$R, 0.75), length(s3$R))  ), 
    lower = c( rep(quantile(s1$R, 0.25), length(s1$R)), 
               rep(quantile(s2$R, 0.25), length(s2$R)), 
               rep(quantile(s3$R, 0.25), length(s3$R))  )	)


p2.6 <- ggplot(df, aes(x = sat, y = mean, colour = sat, fill = sat)) + 
    geom_violin(adjust = 2) + 
    scale_y_continuous(breaks = c(0, 25, 50, 75, 100), limits = c(0, 100), 
        labels = c(0, 25, 50, 75, "")) + 
    scale_fill_manual(values = fill, guide = F) + scale_colour_manual(values = colour, guide = F) + 
    xlab("") + ylab(expression(paste("Reproduction number, ", R[0], sep = ""))) + 
    theme_bw() + 
    #scale_y_continuous(limits = c(0, 35), 
    #                   breaks = c(0, 5, 10, 15, 20, 25, 30, 35)) + 
    theme(axis.line.x = element_line(colour= "black"),
          axis.line.y = element_line(colour= "black"),
          axis.title.x= element_text(size = 15, vjust=-0.15), 
          axis.title.y = element_text(size = 15, vjust= 0.8),
          axis.text.x = element_text(size = 12, vjust=-0.05),
          axis.text.y = element_text(size = 12),
          legend.position=c(0.9, 0.9),
          legend.title = element_blank(),
          legend.key = element_blank(),
          legend.text = element_text(size = 14),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(), axis.line = 
              element_line(colour= "black")  ) + 
    geom_point(data = df, aes(x = sat, y = median), colour = "black") + 
    geom_linerange(data = df, aes(ymin = lower, ymax = upper))    


# put together
##############################################
multiplot(p2.1, p2.4, p2.2, p2.5, p2.3, p2.6, cols=3)

