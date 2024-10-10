import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MaterialModule } from '../material';
import { DirectoryTreeComponent } from './tree/tree.component';
import { CodeEditorComponent, TerminalComponent } from './code-editor/code-editor.component';
import { MonacoEditorModule } from 'ngx-monaco-editor';
import { FormsModule } from '@angular/forms';
import { LocationsTreeComponent, RuleTreeComponent } from './tree/codeql-tree';
import { DataFlowTreeComponent } from './data-flow-tree/data-flow-tree.component';
import { MatTreeModule } from '@angular/material/tree';
import { CdkTreeModule } from '@angular/cdk/tree';


@NgModule({
    declarations: [DirectoryTreeComponent, CodeEditorComponent, RuleTreeComponent, LocationsTreeComponent, TerminalComponent, DataFlowTreeComponent],
    imports: [CommonModule, FormsModule, MaterialModule, MonacoEditorModule.forRoot(), MatTreeModule, CdkTreeModule,],
    exports: [DirectoryTreeComponent, CodeEditorComponent, RuleTreeComponent, LocationsTreeComponent,TerminalComponent, DataFlowTreeComponent],
})
export class SharedComponentsModule { }
