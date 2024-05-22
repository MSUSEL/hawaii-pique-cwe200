import { Component} from '@angular/core';
import {  CVEUtilService } from './cwe-util.service';
import {MatDialog} from '@angular/material/dialog';
import { SaveFileDialogComponent } from './dialogs/dialogs.component';
import { SettingsDialogComponent } from './dialogs/settings.dialog.component';
import {ConfirmationDialogComponent} from '../home/page-header/dialogs/confirmation-dialog.component';

import { ItemFlatNode } from '../shared-components/tree/tree.component';
import { FilesService } from '../Services/fileService';

declare var $: any;
@Component({
    selector: 'app-cwe',
    templateUrl: './main.component.html',
})
export class AttackSurfaceComponent {
    SearchTerm: string = '';
    constructor(
        public utilService: CVEUtilService,
        public fileService:FilesService,
        public dialog: MatDialog
    ) {}

    openDialog(): void {
        var previousName:string=this.utilService.ProjectName;
        const dialogRef = this.dialog.open(SaveFileDialogComponent,{
            height: '320px',
            width: '500px',
            position:{
                top:'100px'
            }
        });

        dialogRef.afterClosed().subscribe(result => {
            if(result){
                this.utilService.export();
            }else{
                if(this.utilService.ProjectName.length==0 && result!=undefined){
                    alert("Invalid Name")
                }
                this.utilService.ProjectName=previousName;
            }
        });
    }

    openSettingsDialog() : void {
        const dialogRef = this.dialog.open(SettingsDialogComponent,{
            height: '320px',
            width: '500px',
            position:{
                top:'100px'
            }
        });
    }


}
