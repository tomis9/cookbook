FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install software-properties-common -y  # for adding R 3.5 repo, 
# some packages require R version higher than default 3.2
RUN apt-get update

RUN add-apt-repository -y "ppa:marutter/rrutter3.5"  # adding R 3.5 version
RUN apt-get update

RUN apt-get install r-base -y

RUN Rscript -e "sessionInfo()"

RUN apt-get install libssl-dev -y  # for R httr package (dependency of tidyverse)
RUN apt-get install libcurl4-openssl-dev -y  # for R httr package (dependency of tidyverse)
RUN apt-get install libxml2-dev -y  # for R rvest package (dependency of tidyverse)

RUN Rscript -e 'install.packages(c("lubridate", "dplyr", "reshape2", "car", "data.table", "tidyverse", "ggplot2", "plotly", "sqldf", "blogdown", "maps", "reticulate", "futile.logger", "rpart.plot", "devtools", "class", "e1071", "mltools", "randomForest", "xgboost", "caret", "kernlab", "ROCR"), repos="http://cran.us.r-project.org")'

RUN Rscript -e 'devtools::install_github("tomis9/decisionTree")'
RUN Rscript -e 'devtools::install_github("vqv/ggbiplot")'

RUN apt-get install python3-pip -y

RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 show setuptools
RUN pip3 install --upgrade setuptools
RUN pip3 install sklearn
RUN pip3 install xgboost
RUN pip3 install tensorflow
RUN pip3 install pandas

RUN apt-get install pandoc -y

RUN Rscript -e "blogdown::install_hugo(version = '0.52')"

RUN pip3 install matplotlib
RUN pip3 install torch==1.0.0

RUN mkdir -p /cookbook/hugo
WORKDIR /cookbook/hugo

CMD Rscript -e 'blogdown::build_site()'
