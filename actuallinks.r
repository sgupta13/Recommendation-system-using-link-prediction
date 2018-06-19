#reading the data into a dataframe
linksdata <- read.table(file = "groundt.txt")
dim(linksdata)
colnames(linksdata) <- c("node1","node2","date")
linksdata<-data.frame(linksdata$node1,linksdata$node2)
dim(linksdata)
zz <- file("groundtruths3.txt", open = "wt")
print("Actual Links in future")
sink(zz)
for(i in 1:594329)
{
  if(linksdata[i,][1] <= 10 || linksdata[i,][2] <= 10)
  {
      a = linksdata[i,][1]
      b = linksdata[i,][2]
      output <- paste(a, b)
      print(output) 
      sink(zz)
  }
}
    


