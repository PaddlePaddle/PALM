
#!/bin/sh
if [[ $# != 1 ]]; then
    echo "usage: bash recover_params.sh <params_dir>"
    exit 1
fi

rm $1/backbone-*
mv $1/.palm.backup/* $1
rm -rf $1/.palm.backup

