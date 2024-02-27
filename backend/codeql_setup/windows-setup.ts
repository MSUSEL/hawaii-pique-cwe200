import { execSync } from 'child_process';

// Windows-specific setup commands
console.log('Running Codeql setup for Windows...');

// Clone the codeql starter workspace with the name codeql
try {
    execSync('git clone --recurse-submodules https://github.com/github/vscode-codeql-starter.git codeql', { stdio: 'inherit' });

} catch (error) {
    console.log('Ignore this error it\'s just related to JS and Csharp dependencies, proceeding to the the next step')
}

// Copy over the queries from codel queries into codeql\\codeql-custom-queries-java
try {
    execSync('xcopy /E /y "codeql queries" codeql\\codeql-custom-queries-java', { stdio: 'inherit' });
} catch (error) {
    console.error('Error occurred during Windows setup:', error);
}
console.log('Windows setup completed.');
