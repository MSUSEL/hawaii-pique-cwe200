import { Injectable } from '@angular/core';
import { ItemFlatNode } from './tree/tree.component';
import { FilesService } from '../Services/fileService';
import { BehaviorSubject } from 'rxjs';

export class SourceFile {
    name: string = '';
    fullPath: string = '';
    code: string = '';
    editorInstance: any; // Add editorInstance to track the editor instance
    constructor(name: string = '', fullPath: string = '', code: string = '', editorInstance: any = null) {
        this.code = code;
        this.name = name;
        this.fullPath = fullPath;
        this.editorInstance = editorInstance; // Track the editor instance
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
        let line = node.region?.startLine;
        let column = node.region?.startColumn;

        console.log('Node clicked:', node); // Log the clicked node for debugging

        if (file) {
            this.activeFile = file;
            this.fileChangedEventEmitter();
            if (line && column) {
                this.scrollToLine(line, column); // Scroll to the specified line and column
            }
        } else {
            this.fileService.getFileContents({ filePath: normalizedFullPath }).subscribe((response) => {
                this.activeFile = new SourceFile(node.name, normalizedFullPath, response.code);
                this.openedFiles.push(this.activeFile);
                this.fileChangedEventEmitter();
                if (line && column) {
                    this.scrollToLine(line, column); // Scroll after loading the file
                }
            });
        }
    }

    // Method to emit the active file change event
    fileChangedEventEmitter() {
        this.activeFileChange.next(this.activeFile);
    }

    // Method to scroll to a specific line and column in the editor
    scrollToLine(line: number, column: number) {
        if (this.activeFile && this.activeFile.editorInstance) {
            this.activeFile.editorInstance.revealPositionInCenter({ lineNumber: line, column: column });
            this.activeFile.editorInstance.setPosition({ lineNumber: line, column: column });
        }
    }

    // Method to set the editor instance once the editor is initialized
    setEditorInstance(filePath: string, editorInstance: any) {
        const file = this.openedFiles.find((f) => f.fullPath === filePath);
        if (file) {
            file.editorInstance = editorInstance; // Attach the editor instance to the file
        }
    }

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
