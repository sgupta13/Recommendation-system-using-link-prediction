#Hits Algorithm
#Implementing HITS algorithm
kmax <- 6
HITS<-function(g,k)
{ 
  adj <- get.adjacency(g, sparse=F) 
  nodes <- dim(adj)[1] 
  auth <- c(rep(1,nodes)) 
  hub <- c(rep(1,nodes)) 
  for(i in 1:k){ 
    t_adj <- t(adj) 
    auth <- t_adj%*%hub 
    hub <- adj%*%auth 
    sum_sq_auth <- sum(auth*auth) 
    sum_sq_hub <- sum(hub*hub) 
    auth <- auth/sqrt(sum_sq_auth) 
    hub <- hub/sqrt(sum_sq_hub) 
  } 
  result <- c(auth,hub)   
  return(result) 
}

op <- HITS(graph,kmax)
write.table(op, "hits.txt")