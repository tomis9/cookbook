#!/bin/bash

#http://ryanstutorials.net/bash-scripting-tutorial/bash-variables.php

# najprostszy program w bashu
# po zapisaniu programu trzeba zmienić uprawnienia za pomocą 
# chmod +x hello_world.sh
# potem program uruchamiany jest standardowo za pomocą ./
STRING="Hello World"
echo $STRING

# teraz trochę utrudnienia: 
# zmienna globalna (i od razu funkcja):
VAR="global variable"
function bash {
    # zmienna lokalna - istnieje tylko w tej funkcji
    local VAR="local variable"
    echo $VAR
}
bash  # wyświetli "local variable"
echo $VAR  # wyświetli "global variable"


