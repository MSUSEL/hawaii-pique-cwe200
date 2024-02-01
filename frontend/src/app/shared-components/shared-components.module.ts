import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MaterialModule } from '../material';
import { DirectoryTreeComponent } from './tree/tree.component';
import { CodeEditorComponent, TerminalComponent } from './code-editor/code-editor.component';
import { MonacoEditorModule } from 'ngx-monaco-editor';
import { FormsModule } from '@angular/forms';
import { LocationsTreeComponent, RuleTreeComponent } from './tree/codeql-tree';

@NgModule({
    declarations: [DirectoryTreeComponent, CodeEditorComponent, RuleTreeComponent, LocationsTreeComponent,TerminalComponent],
    imports: [CommonModule, FormsModule, MaterialModule, MonacoEditorModule.forRoot()],
    exports: [DirectoryTreeComponent, CodeEditorComponent, RuleTreeComponent, LocationsTreeComponent,TerminalComponent],
})
export class SharedComponentsModule { }
