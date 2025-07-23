import { Component, ViewChild, ElementRef, OnInit } from '@angular/core';
import { EditorService, SourceFile } from '../editor-service';

import * as MEditor from 'monaco-editor';
import { ChatGptService } from 'src/app/Services/chatgpt.service';
import { SocketService } from 'src/app/Services/socket-service.service';

declare const monaco: any;

@Component({
    selector: 'app-code-editor',
    templateUrl: './code-editor.component.html',
    styleUrls: ['./code-editor.component.scss'],
})
export class CodeEditorComponent implements OnInit {
    view: string = 'code';
    decorations: string[] = [];
    monacoEditor: MEditor.editor.ICodeEditor = null;
    fileDecorations: any[] = [];

    editorOptions = {
        theme: 'vs-light',
        language: 'java',
        automaticLayout: true,
        scrollBeyondLastLine: false,
    };
    chatGptOptions = { theme: 'vs-light', language: 'text' };

    code: string = 'function x() {\nconsole.log("Hello world!");\n}';
    filePath: string = '';
    chatGptResponse: string = '';

    constructor(
        public editorService: EditorService,
        private chatGptService: ChatGptService,
        private socketService: SocketService
    ) {
        this.editorService.activeFileChange.subscribe((file: SourceFile) => {
            if (file) {
                this.filePath = this.getDisplayPath(file.fullPath);
                this.view = 'code';
                this.chatGptResponse = '';

                // Apply highlighting once the file is active and ready
                if (file.startLine && file.startColumn && file.endLine && file.endColumn && this.monacoEditor) {
                    this.highlightLine(file.startLine, file.startColumn, file.endLine, file.endColumn);
                }
            }
        });
    }

    ngOnInit(): void {}

    // Method to highlight a specific line in the editor
    highlightLine(startLine: number, startColumn: number, endLine: number, endColumn: number) {
        console.log('Highlighting line:', startLine, startColumn, endLine, endColumn);

        if (this.monacoEditor) {
            const model = this.monacoEditor.getModel();
            const totalLines = model.getLineCount();

            if (startLine > totalLines || endLine > totalLines || startLine < 1 || endLine < 1) {
                console.error('Line numbers out of bounds');
                return;
            }

            this.monacoEditor.revealLineInCenter(startLine);

            const range = new monaco.Range(startLine, startColumn, endLine, endColumn);
            const decoration = {
                range: range,
                options: {
                    inlineClassName: 'highlight-line',
                },
            };

            this.decorations = this.monacoEditor.deltaDecorations(this.decorations, [decoration]);

            console.log('Applied decorations:', this.decorations);
        }
    }

    // Handle Monaco editor initialization
    onInitEditor(event: any) {
        this.monacoEditor = event;
        this.editorService.setEditorInstance(this.editorService.activeFile.fullPath, this.monacoEditor);
        
        // Highlight the line once the editor is initialized and the file is active
        if (this.editorService.activeFile && this.editorService.activeFile.startLine && this.editorService.activeFile.startColumn) {
            this.highlightLine(
                this.editorService.activeFile.startLine, 
                this.editorService.activeFile.startColumn, 
                this.editorService.activeFile.endLine, 
                this.editorService.activeFile.endColumn
            );
        }
    }

    onCodeChange(value: string) {
        const model: MEditor.editor.ITextModel = this.monacoEditor.getModel();
        this.highlightLine(
            this.editorService.activeFile.startLine, 
            this.editorService.activeFile.startColumn, 
            this.editorService.activeFile.endLine, 
            this.editorService.activeFile.endColumn
        );
    }

    onFileTabSelected(item: SourceFile) {
        this.editorService.activeFile = item;
        this.editorService.fileChangedEventEmitter();
    }

    closeFile(item: SourceFile) {
        var index = this.editorService.openedFiles.findIndex(
            (file) => file == item
        );
        this.editorService.openedFiles.splice(index, 1);
        if (this.editorService.activeFile == item) {
            if (this.editorService.openedFiles.length) {
                this.editorService.activeFile = this.editorService.openedFiles[0];
            } else {
                this.editorService.activeFile = new SourceFile();
            }
            this.editorService.fileChangedEventEmitter();
        }
    }

    gerChatGptResponse(model: string) {
        this.chatGptResponse = '';
        this.chatGptService
            .queryChatGpt({
                model: model,
                filePath: this.editorService.activeFile.fullPath,
            })
            .subscribe((response) => {
                this.chatGptResponse = response.value;
            });
    }

    // Helper to clean up the file path for display purposes
    getDisplayPath(fullPath: string): string {
        // Remove the Files/ prefix if it exists
        let displayPath = fullPath;
        if (displayPath.startsWith('Files/') || displayPath.startsWith('Files\\')) {
            displayPath = displayPath.replace(/^Files[/\\]/, '');
        }
        
        // Split the path to handle duplicate project names
        const pathComponents = displayPath.split(/[/\\]+/);
        
        if (pathComponents.length > 1) {
            const projectName = pathComponents[0];
            
            // Check if the second component is the same as the project name (duplicate)
            if (pathComponents[1] === projectName) {
                // Remove the duplicate project name
                pathComponents.splice(1, 1);
                displayPath = pathComponents.join('/');
            }
        }
        
        return displayPath;
    }
}

@Component({
    selector: 'app-editor-terminal',
    templateUrl: './terminal.component.html',
})
export class TerminalComponent implements OnInit {
    mode: boolean = true;
    counter = 0;

    constructor(public socketService: SocketService) { }

    @ViewChild('term', { static: true }) terminal: ElementRef;

    ngOnInit() {
        // Terminal logic can be added here if needed
    }
}
