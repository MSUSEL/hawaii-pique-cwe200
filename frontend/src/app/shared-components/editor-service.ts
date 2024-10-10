import { Injectable } from '@angular/core';
import { ItemFlatNode } from './tree/tree.component';
import { FilesService } from '../Services/fileService';
import { BehaviorSubject } from 'rxjs';

export class SourceFile {
    name: string = '';
    fullPath: string = '';
    code: string = '';
    editorInstance: any; // Track the editor instance
    startLine?: number;
    startColumn?: number;
    endLine?: number;
    endColumn?: number;    

    constructor(name: string = '', fullPath: string = '', code: string = '', editorInstance: any = null, 
        startLine: number = null, startColumn: number = null, endLine: number = null, endColumn: number = null) {
        this.name = name;
        this.fullPath = fullPath;
        this.code = code;
        this.editorInstance = editorInstance;
        this.startLine = startLine;
        this.startColumn = startColumn;
        this.endLine = endLine;
        this.endColumn = endColumn;
    }
}

@Injectable({
    providedIn: 'root',
})
export class EditorService {
    activeFileChange = new BehaviorSubject<SourceFile>(null);
    activeFile: SourceFile = new SourceFile();
    openedFiles: SourceFile[] = [];
    rulesTree: any[] = [];
    locationsTree: any[] = [];

    constructor(private fileService: FilesService) {}

    // Method to find the file in the opened files or open it if it's not
    findFile(node: ItemFlatNode) {
        const normalizedFullPath = this.correctPath(node.fullPath);

        let file = this.openedFiles.find((item) => item.fullPath == normalizedFullPath);
        let startLine = node.region?.startLine;
        let startColum = node.region?.startColumn;
        let endLine = node.region?.endLine;
        let endColumn = node.region?.endColumn;

        console.log('Editor Service', node); // Log the clicked node for debugging

        if (file) {
            // Update the file with line and column information
            file.startLine = startLine;
            file.startColumn = startColum;
            file.endLine = endLine;
            file.endColumn = endColumn;
            this.activeFile = file;
            this.fileChangedEventEmitter();
        } else {
            this.fileService.getFileContents({ filePath: normalizedFullPath }).subscribe((response) => {
                // Create a new file and pass the line and column for highlighting
                this.activeFile = new SourceFile(node.name, normalizedFullPath, response.code, null, startLine, startColum, endLine, endColumn);
                this.openedFiles.push(this.activeFile);
                this.fileChangedEventEmitter();
            });
        }
    }

    // Method to emit the active file change event
    fileChangedEventEmitter() {
        this.activeFileChange.next(this.activeFile);
    }

    // Method to set the editor instance once the editor is initialized
    setEditorInstance(filePath: string, editorInstance: any) {
        const file = this.openedFiles.find((f) => f.fullPath === filePath);
        if (file) {
            file.editorInstance = editorInstance; // Attach the editor instance to the file
        }
    }

    // Correct the path to maintain consistency
    correctPath(filePath: string): string {
        // Replace backslashes with forward slashes
        filePath = filePath.replace(/\\/g, '/');

        // Extract project name dynamically using the first component of the path
        const pathComponents = filePath.split(/[/\\]+/);
        const projectName = pathComponents[0];  // Always use the first component as the project name

        if (!filePath.startsWith('Files/')) {
            filePath = `Files/${projectName}/${filePath}`;
        }

        return filePath;
    }
}
