import { Component, OnInit } from '@angular/core';
import { MatDialogRef } from '@angular/material/dialog';
import { CVEUtilService } from '../cwe-util.service';
import { ChatGptService } from 'src/app/Services/chatgpt.service';

@Component({
    selector: 'help-dialogs',
    templateUrl: './help-dialog.component.html',
})
export class HelpDialogComponent implements OnInit {

    constructor(
        public dialogRef: MatDialogRef<HelpDialogComponent>,
        public utilService: CVEUtilService,
        public chatGptService: ChatGptService
    ) {}

    ngOnInit(): void {
        
    }

    clearTokenOnFocus() {
    }

    onSaveClick(): void {
        
    }

    onNoClick(): void {
        this.dialogRef.close();
    }
}
