import * as path from 'path';
import { execSync } from 'child_process';
import * as fs from 'fs';

// Detect the platform
console.log(`Detected platform: ${process.platform}`);
console.log('Running CodeQL setup on Linux...');

// Define paths
const codeqlPath = path.join('codeql');
const qlSubmodulePath = path.join('codeql', 'ql');
const customQueriesPath = path.join('codeql', 'codeql-custom-queries-java');
const sourceQueriesPath = path.join('codeql queries');

// Clone the CodeQL starter workspace if it doesn't exist
try {
    if (!fs.existsSync(codeqlPath)) {
        execSync(`git clone --recurse-submodules https://github.com/github/vscode-codeql-starter.git ${codeqlPath}`, { stdio: 'inherit' });
        console.log('CodeQL repository cloned successfully.');
    } else {
        console.log('CodeQL repository already exists. Skipping clone.');
    }

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
    console.error('Error during Git operations:', error);
}

// Remove the example.ql file before copying the queries
try {
    const exampleQueryPath = path.join(customQueriesPath, 'example.ql');
    if (fs.existsSync(exampleQueryPath)) {
        fs.unlinkSync(exampleQueryPath);
        console.log('Removed example.ql file successfully.');
    } else {
        console.log('example.ql file not found, no need to remove.');
    }
} catch (error) {
    console.error('Error while trying to remove example.ql:', error);
}

// Copy over the queries from 'codeql queries' into 'codeql/codeql-custom-queries-java'
try {
    if (fs.existsSync(sourceQueriesPath)) {
        execSync(`cp -r "${sourceQueriesPath}/." "${customQueriesPath}/"`, { stdio: 'inherit' });
        console.log('Queries copied successfully.');
    } else {
        console.log(`Source queries directory '${sourceQueriesPath}' not found.`);
    }
} catch (error) {
    console.error('Error occurred during query copying:', error);
}

console.log('Setup completed.');
