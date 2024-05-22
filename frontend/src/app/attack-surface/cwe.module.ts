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
import { ConfirmationDialogComponent } from '../home/page-header/dialogs/confirmation-dialog.component';



@NgModule({
    declarations: [
        AttackSurfaceComponent,
        PageHeaderComponent,
        MainComponent,
        SaveFileDialogComponent,
        CodeQlComponent,
        ConfirmationDialogComponent
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
