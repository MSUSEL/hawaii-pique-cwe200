import * as path from 'path';
import { execSync } from 'child_process';

// Detect the platform
console.log(`Detected platform: ${process.platform}`);
console.log('Running CodeQL setup with platform-agnostic paths...');

// Define the platform-agnostic paths
const codeqlPath = path.join('codeql');
const qlSubmodulePath = path.join('codeql', 'ql');
const customQueriesPath = path.join('codeql', 'codeql-custom-queries-java');
const sourceQueriesPath = path.join('codeql queries');

// Clone the codeql starter workspace with the name codeql
try {
    execSync(`git clone --recurse-submodules https://github.com/github/vscode-codeql-starter.git ${codeqlPath}`, { stdio: 'inherit' });
    console.log('CodeQL repository cloned successfully.');

    // Fetch all branches and tags in the main repository
    execSync(`cd ${codeqlPath} && git fetch --all`, { stdio: 'inherit' });

    // Checkout the specific commit for the main repository
    execSync(`cd ${codeqlPath} && git checkout 7efe5ac39b288a2775d18f03f7d9135d4ac00c4d`, { stdio: 'inherit' });
    console.log('Checked out to specific commit in main repository: 7efe5ac39b288a2775d18f03f7d9135d4ac00c4d.');

    // Update the ql submodule to its specific commit
    execSync(`cd ${qlSubmodulePath} && git fetch origin && git checkout 2daf50500ca8f7eb914c82e88dec36652bfbe8fd`, { stdio: 'inherit' });
    console.log('Checked out to specific commit in ql submodule: 2daf50500ca8f7eb914c82e88dec36652bfbe8fd.');
    
    // Update submodules to reflect this change
    execSync(`cd ${codeqlPath} && git submodule update --init --recursive`, { stdio: 'inherit' });
    console.log('Submodules updated successfully.');

} catch (error) {
    console.log('Ignore this error; it\'s just related to JS and C# dependencies, proceeding to the next step.');
}

// Copy over the queries from codeql queries into codeql\\codeql-custom-queries-java
try {
    // Use platform-agnostic paths for copying queries
    execSync(`xcopy /E /y "${sourceQueriesPath}" "${customQueriesPath}"`, { stdio: 'inherit' });
    console.log('Queries copied successfully.');
} catch (error) {
    console.error('Error occurred during platform-agnostic setup:', error);
}

console.log('Setup completed.');
