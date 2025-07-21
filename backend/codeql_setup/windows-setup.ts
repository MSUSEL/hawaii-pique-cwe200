import * as path from 'path';
import { execSync } from 'child_process';
import * as fs from 'fs';

// Detect the platform
console.log(`Detected platform: ${process.platform}`);
console.log('Running CodeQL setup with platform-agnostic paths...');

// Define the platform-agnostic paths
const codeqlPath = path.join('codeql');
const qlSubmodulePath = path.join('codeql', 'ql');
const customQueriesPath = path.join('codeql', 'codeql-custom-queries-');
const sourceQueriesPath = path.join('codeql queries');
const supportedLanguages = ['java', 'python'];

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
    execSync(`cd ${qlSubmodulePath} && git fetch origin && git checkout e27d8c16729588259f8143c7ed4569d517b0de10`, { stdio: 'inherit' });
    console.log('Checked out to specific commit in ql submodule: e27d8c16729588259f8143c7ed4569d517b0de10.');

    // Update submodules to reflect this change
    execSync(`cd ${codeqlPath} && git submodule update --init --recursive`, { stdio: 'inherit' });
    console.log('Submodules updated successfully.');

} catch (error) {
    console.log('Ignore this error; it\'s just related to JS and C# dependencies, proceeding to the next step.');
}

// Remove the example.ql file before copying the queries

for (const language of supportedLanguages) {
    const thisCustomQueriesPath = customQueriesPath + language;
    const thisSourceQueriesPath = path.join(sourceQueriesPath, language);

    try {
        const exampleQueryPath = path.join(thisCustomQueriesPath, 'example.ql');

        if (fs.existsSync(exampleQueryPath)) {
            fs.unlinkSync(exampleQueryPath);
            console.log('Removed example.ql file successfully.');
        } else {
            console.log('example.ql file not found, no need to remove.');
        }
    } catch (error) {
        console.error('Error while trying to remove example.ql:', error);
    }

    // Copy over the queries from codeql queries into codeql\\codeql-custom-queries-java
    try {
        // Use platform-agnostic paths for copying queries
        execSync(`xcopy /E /y "${thisSourceQueriesPath}" "${thisCustomQueriesPath}"`, { stdio: 'inherit' });
        console.log('Queries copied successfully.');
    } catch (error) {
        console.error('Error occurred during platform-agnostic setup:', error);
    }
}

console.log('Codeql Setup completed.');
