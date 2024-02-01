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
export class CodeEditorComponent  implements OnInit{
    view: string = 'code';
    decorations: string[] = [];
    monacoEditor: MEditor.editor.ICodeEditor = null;
    fileDocarations: any[] = [];

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
        private socketService:SocketService
    ) {
        this.editorService.activeFileChange.subscribe((file) => {
            if (file) {
                this.filePath = file.fullPath.replace('Files\\', '');
                this.view = 'code';
                this.chatGptResponse = '';
            }
        });
    }
    ngOnInit(): void {

    }

    getFileDecorations() {
        this.fileDocarations = [];
        var file = this.editorService.locationsTree.find(
            (item) => item.fullPath == this.editorService.activeFile.fullPath
        );
        if (file) {
            this.fileDocarations = file.files;
        }
    }

    highLightEditor() {
        this.getFileDecorations();
        this.decorations = this.monacoEditor.deltaDecorations(
            this.decorations,
            []
        );
        var options = {
            className: 'bg-creamy radius-25',
        };
        var decorationRanges = [];
        this.fileDocarations.forEach((file) => {
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
        const range = new monaco.Range(1, 1, 4, 5);
        this.decorations = this.monacoEditor.deltaDecorations(
            [],
            decorationRanges
        );

        this.updateHoverProvider(this.monacoEditor.getModel().getValue());
    }
    hoverProvider:any=null;
    updateHoverProvider(content: string) {
        if(this.hoverProvider){
            this.hoverProvider.dispose()
        }
        this.hoverProvider=monaco.languages.registerHoverProvider('java', {
            provideHover: (model, position) => {
                for (let range of this.fileDocarations) {
                    if (
                        (position.lineNumber >= range.region.startLine && position.column >= range.region.startColumn)&&
                        (position.lineNumber <= range.region.endLine && position.column <= range.region.endColumn)
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

    onCodeChange(value: string) {
        const model: MEditor.editor.ITextModel = this.monacoEditor.getModel();
        this.highLightEditor();
    }

    onInitEditor(event: any) {
        this.monacoEditor = event;
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
                this.editorService.activeFile =
                    this.editorService.openedFiles[0];
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
    mode:boolean=true;
    counter=0;
    constructor(
        public socketService:SocketService
    ){}
    @ViewChild('term', { static: true }) terminal: ElementRef;
    ngOnInit(){

    }
}
