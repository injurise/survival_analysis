
library(coxed)

simdata <- sim.survdata(N=1000, T=356, num.data.frames=1)
quant  = unname(quantile(simdata$data$y, seq(0.1, 0.9, by=0.1)))
pseudo <- pseudosurv(time=simdata$data$y,event=simdata$data$failed,tmax=quant)
surv_pdeudo_data <- NULL
for(it in 1:length(pseudo$time)){
  surv_pdeudo_data <- rbind(surv_pdeudo_data,cbind(simdata$data,pseudo = pseudo$pseudo[,it],
                                                   tpseudo = pseudo$time[it],id=1:nrow(simdata$data)))
}
surv_pdeudo_data <- b[order(b$id),]

write.table(surv_pdeudo_data, 
            file="/Users/alexandermollers/Documents/GitHub/survival_analysis/data/surv_pseudo_data.csv", 
            sep = ',', row.names = FALSE, col.names = TRUE)