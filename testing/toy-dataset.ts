import * as fs from 'fs';
import * as path from 'path';

const PROJECT = "CWEToyDataset";
const CHATGPTPATH = `Files/${PROJECT}/data.json`;
const CODEQLPATH = `Files/${PROJECT}/results.sarif`;
const SRCCODEPATH = `Files/${PROJECT}`;
const CWESPATH = `Files/${PROJECT}/src/main/java/com/mycompany/app`;

async function main() {
    process.chdir("backend/");
    const javaFiles = getJavaFiles(SRCCODEPATH);
    const chatgptResults = readData(CHATGPTPATH);
    const cwes = getDirectoriesInDir(CWESPATH);

    analyzeChatgptResults(await chatgptResults, await javaFiles);

    console.log(1);
}

function readData(path: string): Promise<any> {
    return new Promise((resolve, reject) => {
        fs.readFile(path, { encoding: 'utf-8' }, (err, data) => {
            if (err) {
                reject(err);
                return;
            }
            if (path.endsWith(".json")) {
                resolve(JSON.parse(data));
            } else if (path.endsWith(".sarif")) {
                // Add logic for parsing SARIF files if needed
                resolve(data); // Placeholder
            }
        });
    });
}

function getJavaFiles(path: string): Promise<string[]> {
    return new Promise((resolve, reject) => {
        let javaFiles: string[] = [];
        fs.readdir(path, { withFileTypes: true }, (err, files) => {
            if (err) {
                reject(err);
                return;
            }
            files.forEach(file => {
                if (file.isFile() && file.name.endsWith('.java')) {
                    javaFiles.push(path + '/' + file.name);
                }
            });
            resolve(javaFiles);
        });
    });
}

function getDirectoriesInDir(directoryPath: string): string[] {
    return fs.readdirSync(directoryPath, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory())
        .map(dirent => dirent.name);
}

function analyzeChatgptResults(chatgptResults: any[], javaFiles: string[]) {
    let truePositives = 0;
    let falsePositives = 0;
    let trueNegatives = 0;
    let falseNegatives = 0;

    // TypeScript doesn't have a direct equivalent to Python's defaultdict,
    // so you might need a custom implementation or workaround
    let cweSpecificResults: any = {};

    javaFiles.forEach(file => {
        const split = file.split("\\");
        const fileName = split[split.length - 1];
        const cwe = split[split.length - 2];

        chatgptResults.forEach(res => {
            if (res['key'] === fileName) {
                const val = res['value'];

                // Implement vulnerability check logic as needed
                if (fileName.startsWith("BAD") && hasVulnerability(val)) {
                    truePositives++;
                    // Initialize or append to cweSpecificResults
                    console.log(`True positive on ${fileName} with CWE ${cwe}`);
                }
                // Other conditions as in your Python code
            }
        });
    });

    console.log(truePositives, falsePositives);
}

function hasVulnerability(chatgptResults: any[]): boolean {
    return chatgptResults.length > 0;
}

main();
