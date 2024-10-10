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
                this.filePath = file.fullPath.replace('Files\\', '');
                this.view = 'code';
                this.chatGptResponse = '';

                // Highlight the specific line and column when the file changes
                if (file.startLine && file.startColumn && file.endLine && file.endColumn) { 
                    this.highlightLine(file.startLine, file.startColumn, file.endLine, file.endColumn);
                }
            }
        });
    }

    ngOnInit(): void {}

    getFileDecorations() {
        this.fileDecorations = [];
        var file = this.editorService.locationsTree.find(
            (item) => item.fullPath == this.editorService.activeFile.fullPath
        );
        if (file) {
            this.fileDecorations = file.files;
        }
    }

    highLightEditor() {
        this.getFileDecorations();
        this.decorations = this.monacoEditor.deltaDecorations(
            this.decorations,
            []
        );
        var decorationRanges = [];
        this.fileDecorations.forEach((file) => {
            decorationRanges.push({
                range: new monaco.Range(
                    file.region.startLine,
                    file.region.startColumn,
                    file.region.endLine,
                    file.region.endColumn
                ),
                options: {
                    className: 'bg-creamy radius-25 level-' + file.type,
                },
            });
        });
        this.decorations = this.monacoEditor.deltaDecorations([], decorationRanges);
        this.updateHoverProvider(this.monacoEditor.getModel().getValue());
    }

    hoverProvider: any = null;

    updateHoverProvider(content: string) {
        if (this.hoverProvider) {
            this.hoverProvider.dispose();
        }
        this.hoverProvider = monaco.languages.registerHoverProvider('java', {
            provideHover: (model, position) => {
                for (let range of this.fileDecorations) {
                    if (
                        position.lineNumber >= range.region.startLine &&
                        position.column >= range.region.startColumn &&
                        position.lineNumber <= range.region.endLine &&
                        position.column <= range.region.endColumn
                    ) {
                        return {
                            range: new monaco.Range(
                                range.region.startLine,
                                range.region.startColumn,
                                range.region.endLine,
                                range.region.endColumn
                            ),
                            contents: [
                                {
                                    value: range.name,
                                    isTrusted: true,
                                },
                                {
                                    value: range.message,
                                },
                            ],
                        };
                    }
                }
                return null;
            },
        });
    }

    // Method to highlight a specific line in the editor
    highlightLine(startLine: number, startColumn: number, endLine: number, endColumn: number) {
        console.log('Highlighting line:', startLine, startColumn, endLine, endColumn);
        if (this.monacoEditor) {
            const model = this.monacoEditor.getModel();
            const totalLines = model.getLineCount();

            // Ensure the lines are within bounds
            if (startLine > totalLines || endLine > totalLines || startLine < 1 || endLine < 1) {
                console.error('Line numbers out of bounds');
                return;
            }

            // Create a range using the provided start and end line and column values
            const range = new monaco.Range(startLine, startColumn, endLine, endColumn);

            // Create the decoration using the range and apply the CSS class for the highlight
            const decoration = {
                range: range,
                options: {
                    inlineClassName: 'highlight-line',  // Use a CSS class for the highlight
                },
            };

            // Apply the decoration to highlight the specific range
            this.decorations = this.monacoEditor.deltaDecorations(this.decorations, [decoration]);

            // Log the decorations to see what is applied
            console.log('Applied decorations:', this.decorations);

            // Scroll to the specified start position
            this.monacoEditor.revealPositionInCenter({ lineNumber: startLine, column: startColumn });
        }
    }

    // Handle Monaco editor initialization
    onInitEditor(event: any) {
        this.monacoEditor = event;
        // Set the editor instance in the active file for reference
        this.editorService.setEditorInstance(this.editorService.activeFile.fullPath, this.monacoEditor);
        this.highLightEditor();
    }

    onCodeChange(value: string) {
        const model: MEditor.editor.ITextModel = this.monacoEditor.getModel();
        this.highLightEditor();
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
