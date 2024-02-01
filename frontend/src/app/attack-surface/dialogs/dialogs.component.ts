import { Component, OnInit } from '@angular/core';
import {MatDialogRef} from '@angular/material/dialog';
import { CVEUtilService } from '../cwe-util.service';

@Component({
    selector: 'save-dialogs',
    templateUrl: './save-file-dialog.component.html',
})
export class SaveFileDialogComponent  {
   
        constructor(
          public dialogRef: MatDialogRef<SaveFileDialogComponent>,
          public utilService:CVEUtilService
        ) {}
      
        onNoClick(): void {
          this.dialogRef.close();
        }
      
}
