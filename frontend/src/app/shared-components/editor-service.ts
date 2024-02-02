import { Injectable } from '@angular/core';
import { ItemFlatNode} from './tree/tree.component';
import { FilesService } from '../Services/fileService';
import { BehaviorSubject } from 'rxjs';

export class SourceFile {
    name: string = '';
    fullPath: string = '';
    code: string = '';
    constructor(name: string = '', fullPath: string = '', code: string = '') {
        this.code = code;
        this.name = name;
        this.fullPath = fullPath;
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
    locationsTree:any[]=[];

    constructor(private fileService: FilesService) {}
    findFile(node: ItemFlatNode) {
        var file = this.openedFiles.find(
            (item) => item.fullPath == node.fullPath
        );
        if (file) {
            this.activeFile = file;
            this.fileChangedEventEmitter();
        } else {
            this.fileService
                .getFileContents({ filePath: node.fullPath })
                .subscribe((response) => {
                    this.activeFile = node;
                    this.activeFile.code = response.code;
                    this.openedFiles.push(
                        new SourceFile(node.name, node.fullPath, response.code)
                    );
                    this.fileChangedEventEmitter();
                });
        }
    }

    fileChangedEventEmitter() {
        this.activeFileChange.next(this.activeFile);
    }
}
