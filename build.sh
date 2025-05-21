#!/bin/bash

mkdir -p package

cp -r *.py package/

pip install --target ./package -r requirments.txt

cd package
rm -rf numpy boto3 botocore

zip -r ../package.zip .
cd ..

rm -rf package
