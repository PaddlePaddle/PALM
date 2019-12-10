
#!/bin/sh
if [[ $# != 1 ]]; then
    echo "usage: bash recover_params.sh <params_dir>"
    exit 1
fi

if [[ ! -d $1 ]]; then
    echo "$1 not found."
    exit 1
fi

if [[ ! -f $1/__palmmodel__ ]]; then
    echo "paddlepalm model not found."
    exit 1
fi

echo "recovering..."
if [[ -d $1/params ]]; then
    cd $1/params
else
    cd $1
fi
rm __palm*
mv .palm.backup/__rawmodel__ .
rm -rf .palm.backup
tar -xf __rawmodel__
mv .palm.backup/* .
rm __rawmodel__

rm -rf .palm.backup
cd - >/dev/null

