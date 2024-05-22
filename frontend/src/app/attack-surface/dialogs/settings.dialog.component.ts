import { Component, OnInit } from '@angular/core';
import { MatDialogRef } from '@angular/material/dialog';
import { CVEUtilService } from '../cwe-util.service';
import { ChatGptService } from 'src/app/Services/chatgpt.service';

@Component({
    selector: 'settings-dialogs',
    templateUrl: './settings-dialog.component.html',
})
export class SettingsDialogComponent implements OnInit {
    isLoading: boolean = true;
    placeholderToken: string;

    constructor(
        public dialogRef: MatDialogRef<SettingsDialogComponent>,
        public utilService: CVEUtilService,
        public chatGptService: ChatGptService
    ) {}

    ngOnInit(): void {
        this.chatGptService.getChatGptToken().subscribe({
            next: (response) => {
                this.isLoading = false;
                this.placeholderToken = response.token; // Assuming the token is returned in response.token
                this.chatGptService.ChatGPTToken = response.token;
            },
            error: (error) => {
                this.isLoading = false;
                console.error('Error fetching API token', error);
            }
        });
    }

    clearTokenOnFocus() {
        this.chatGptService.ChatGPTToken = '';
    }

    onSaveClick(): void {
        this.chatGptService.updateChatGptToken(this.chatGptService.ChatGPTToken).subscribe({
            next: (response) => {
                this.dialogRef.close({ token: this.chatGptService.ChatGPTToken, model: this.chatGptService.GPTModel });
            },
            error: (error) => {
                console.error('Error updating API token', error);
            }
        });
    }

    onNoClick(): void {
        this.dialogRef.close();
    }
}
