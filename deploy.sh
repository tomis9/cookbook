#!/usr/bin/env bash
mod=$1
bs=$2

dev='baseURL = ""'
prod='baseURL = "http:\/\/tomis9\.com"'


if [ "$mod" == "dev" ]
then
    if [ "$bs" == "build" ]
    then
        sed -i "1s/.*/$dev/" config.toml
        Rscript -e "blogdown::build_site()"
        sed -i "1s/.*/$prod/" config.toml
    elif [ "$bs" == "serve" ]
    then
        Rscript -e "blogdown::serve_site()"
    else
        echo "You must choose between build and serve"
    fi
elif [ "$mod" == "prod" ]
then
    if [ "$bs" == "build" ]
    then
        Rscript -e "blogdown::build_site()"
    elif [ "$bs" == "serve" ]
    then
        Rscript -e "blogdown::serve_site()"
    else
        echo "You must choose between build and serve"
    fi
else
    echo "First argument:   [dev|prod]"
    echo "Second argument:  [build|serve]"
fi
