#!/bin/bash

set -e

if [[ $# != 1 ]]; then
    echo "Usage: bash download_pretrain.sh <bert|ernie>"
    exit 1
fi

if [[ $1 == 'bert' ]]; then
    name="bert"
    link="https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz"
    packname="uncased_L-24_H-1024_A-16.tar.gz"
    dirname="uncased_L-24_H-1024_A-16"
elif [[ $1 == 'ernie' ]]; then
    name="ernie"
    link="https://ernie.bj.bcebos.com/ERNIE_Large_en_stable-2.0.0.tar.gz"
    packname="ERNIE_Large_en_stable-2.0.0.tar.gz"
else
    echo "$1 is currently not supported."
    exit 1
fi

cd pretrain_model
mkdir $name
cd $name
echo "downloading ${name}..."
wget --no-check-certificate $link
echo "decompressing..."
tar -zxf $packname
rm -rf $packname
if [[ $dirname != "" ]]; then
    mv $dirname/* .
    rm -rf $dirname
fi

cd ../..


