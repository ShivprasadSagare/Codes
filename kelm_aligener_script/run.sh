#! /bin/sh

langs=(bn en gu kn mr ta te)
for lang in ${langs[*]} 
do
    mkdir $lang
    cp $lang* $lang/
done