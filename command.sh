#!/bin/bash
git pull
FrontendPath="./CWE-Frontend";
BackendPath="./cwe-backend";
cd ${FrontendPath}
npm i --force
ng build
cd ../${BackendPath}
npm i --force
nest build
npm run start:dev
