import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MaterialModule } from '../material';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { AttackSurfaceComponent } from './main.component';
import { PageHeaderComponent } from '../home/page-header/page-header.component';
import { MainComponent } from './MainComponents/main.component';
import { SaveFileDialogComponent } from './dialogs/dialogs.component';
import { SharedComponentsModule } from '../shared-components/shared-components.module';
import { CodeQlComponent } from './code-ql/code-ql.component';



@NgModule({
    declarations: [
        AttackSurfaceComponent,
        PageHeaderComponent,
        MainComponent,
        SaveFileDialogComponent,
        CodeQlComponent,
    ],
    imports: [
        CommonModule,
        ReactiveFormsModule,
        FormsModule,
        MaterialModule,
        SharedComponentsModule
    ],
    providers: [

      ],
})
export class AttackSurfaceModule {}
