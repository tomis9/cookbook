#!/usr/bin/env bash
mod=$1
bs=$2

if [ $mod == "dev" ]
then
    if [ $bs == "build" ]
    then
        echo "dev build"
    elif [ $bs == "serve" ]
    then
        echo "dev serve"
    else
        echo "You must choose between build and serve"
    fi
elif [ $mod == "prod" ]
then
    if [ $bs == "build" ]
    then
        echo "dev build"
    elif [ $bs == "serve" ]
    then
        echo "dev serve"
    else
        echo "You must choose between build and serve"
    fi
else
    echo "You must choose between dev and prod."
fi
