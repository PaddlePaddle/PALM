
#!/bin/sh
if [[ $# != 1 ]]; then
    echo "usage: bash convert_params.sh <params_dir>"
    exit 1
fi

echo "converting..."
cd $1
mkdir .palm.backup

for file in $(ls *)
    do cp $file "backbone-"$file; mv $file .palm.backup
done
cd - >/dev/null

echo "done!"

