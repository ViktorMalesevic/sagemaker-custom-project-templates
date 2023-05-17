#!/bin/bash
CONFIG_FILE=$1
ARTIFACT_BUCKET_NAME=$2

ROOT=""
NAME=""
WORKDIR=$(pwd)

rm -r publish
mkdir publish

echo "Parsing components .."
while read -r line; do
    if [[ $line == root:* ]]; then
        echo $line
        ROOT=$(echo $line | sed 's/root://' | sed 's/ //' | tr -d '\"')
    elif [[ $line == component-name:* ]]; then
        echo $line
        NAME=$(echo $line | sed 's/component-name://' | sed 's/ //' | tr -d '\"')
    fi

    if [[ ! -z "$ROOT" ]] && [[ ! -z "$NAME" ]]; then
        echo "Creating folder $ROOT in output artifact folder .."
        mkdir -p $WORKDIR/publish/$ROOT

        echo "switching folder to $ROOT .."
        cd $WORKDIR/$ROOT

        echo "Perform build of $NAME component .."
        gdk component build
        echo "Uploading component to $ARTIFACT_BUCKET_NAME .."
        aws s3 sync ./greengrass-build/artifacts/ s3://$ARTIFACT_BUCKET_NAME/artifacts/greengrass/

        echo "Uploading recipe to output artifact folder .."
        # Keyword URI is used to push to S3 during publish
        # Uri publishes from S3 directly if build folder is empty
        sed -i "s|URI|Uri|g" recipe.json
        # setting build system to zip for all components after build
        # this is done because custom build doesn't allow to publish directly from s3
        jq '.component."'"$NAME"'".build = {   "build_system": "zip"   }' gdk-config.json > tmp && mv tmp gdk-config.json
        cp ./*.json $WORKDIR/publish/$ROOT
        echo "Done."

        ROOT=""
        NAME=""
    fi

done < $CONFIG_FILE

cd $WORKDIR