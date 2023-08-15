import { Component } from '@angular/core';
import { EditorService } from 'src/app/shared-components/editor-service';

@Component({
    selector: 'app-code-ql',
    templateUrl: './code-ql.component.html',
    styleUrls: ['./code-ql.component.scss'],
})
export class CodeQlComponent {
    constructor(public editorService:EditorService){}
}
