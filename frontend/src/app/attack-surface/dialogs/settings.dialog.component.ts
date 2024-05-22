import { Component, OnInit } from '@angular/core';
import {MatDialogRef} from '@angular/material/dialog';
import { CVEUtilService } from '../cwe-util.service';

@Component({
    selector: 'settings-dialogs',
    templateUrl: './settings-dialog.component.html',
})
export class SettingsDialogComponent  {
   
        constructor(
          public dialogRef: MatDialogRef<SettingsDialogComponent>,
          public utilService:CVEUtilService
        ) {}
      
        onNoClick(): void {
          this.dialogRef.close();
        }
      
}
