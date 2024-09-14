const { execSync } = require('child_process');
const os = require('os');

// Function to check the platform and execute the appropriate command
function runPlatformSpecificCommand() {
    const platform = os.platform();
    console.log('Detected platform:', platform);

    try {
        // Run the appropriate platform-specific command
        if (platform === 'win32') {
            execSync('ts-node backend/codeql_setup/windows-setup.ts', { stdio: 'inherit' });
        } else {
            execSync('ts-node backend/codeql_setup/linux-setup.ts', { stdio: 'inherit' });
        }
    } catch (error) {
        console.error('Error occurred while running platform-specific command:', error);
        process.exit(1);
    }
}

// Run the platform-specific command
runPlatformSpecificCommand();
