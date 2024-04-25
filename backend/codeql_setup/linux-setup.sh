#!/bin/bash

# Logging the start of setup
echo "Running Codeql setup..."

# Cloning the codeql starter workspace
echo "Cloning the codeql starter workspace..."
git clone --recurse-submodules https://github.com/github/vscode-codeql-starter.git codeql
if [ $? -eq 0 ]; then
    echo "Repository cloned successfully."
else
    echo "Ignore this error it's just related to JS and C# dependencies, proceeding to the next step."
fi

# Copy over the queries from "codeql queries" into "codeql/codeql-custom-queries-java"
echo "Copying queries..."
cp -r "codeql queries"/* "codeql/codeql-custom-queries-java/"
if [ $? -eq 0 ]; then
    echo "Queries copied successfully."
else
    echo "Error occurred during the setup process."
fi

echo "Setup completed."






# import { execSync } from 'child_process';

# // Linux-specific setup commands
# console.log('Running Codeql setup for Linux...');

# // Clone the codeql starter workspace with the name codeql
# try {
#     execSync('git clone --recurse-submodules https://github.com/github/vscode-codeql-starter.git codeql', { stdio: 'inherit' });
# } catch (error) {
#     console.log('Ignore this error it\'s just related to JS and C# dependencies, proceeding to the next step')
# }

# // Copy over the queries from 'codeql queries' into 'codeql/codeql-custom-queries-java'
# try {
#     execSync('cp -r "codeql queries"/* codeql/codeql-custom-queries-java/', { stdio: 'inherit' });
# } catch (error) {
#     console.error('Error occurred during Linux setup:', error);
# }
# console.log('Linux setup completed.');
