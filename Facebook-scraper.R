## *Scraper for public Facebook pages that contain a specific keyword in their title*
###Author: Chiara Bonacchi
###Project: IARH

# Set working directory. 
setwd("~path")

# Require Rfacebook, devtools and xlsx packages. If you have not installed htenm already, run install.packages() first.
require(Rfacebook)
require(devtools)
require(xlsx)

# Obtain a temporary user access token to access Facebook API from https://developers.facebook.com/tools-and-support/.
token <- "inserttoken"

# Extract all the public Facebook pages (and reltaed information available) that contain the work "Medieval in their title".
Medieval <- searchPages(string="Medieval", token=token, n=1000)

# Save the list of public Facebook pages containing the world "Medieval" in an excel file.
write.xlsx(Medieval, "medieval_Facebook-pages.xlsx")

# Extract each of the entries in the list and extract their content, with a max of 10,000 post per page and save the content of each of them separately.
usernames <- (medieval$username)
for i in usernames {
	a <- getPage(page=i, token=token, n=1000, feed=TRUE)
	write.xlsx(a, paste(i,".xlsx",sep="")) 
}

