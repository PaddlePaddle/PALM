
#!/bin/sh
if [[ $# != 1 ]]; then
    echo "usage: bash convert_params.sh <params_dir>"
    exit 1
fi

if [[ -f $1/__palminfo__ ]]; then
    echo "already converted."
    exit 0
fi

echo "converting..."
if [[ -d $1/params ]]; then
    cd $1/params
else
    cd $1
fi

mkdir .palm.backup

for file in $(ls *)
    do cp $file .palm.backup; mv $file "__paddlepalm_"$file
done
tar -cf __rawmodel__ .palm.backup/*
rm .palm.backup/*
mv __rawmodel__ .palm.backup
# find . ! -name '__rawmodel__' -exec rm {} +
tar -cf __palmmodel__ __paddlepalm_*
touch __palminfo__
ls __paddlepalm_* > __palminfo__
rm __paddlepalm_*

cd - >/dev/null

echo "done!"

