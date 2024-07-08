import { ChangeDetectorRef, Component, Inject, OnInit, OnDestroy } from '@angular/core';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { SocketService } from '../../../Services/socket-service.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-confirmation-dialog',
  templateUrl: './confirm-chatgpt-cost-dialog.component.html',
})
export class ConfirmationDialogComponent implements OnInit, OnDestroy {
  progress: number = 0;
  private progressSubscription: Subscription;

  constructor(
    public dialogRef: MatDialogRef<ConfirmationDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: any,
    private socketService: SocketService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.socketService.socketConnect();
    this.progressSubscription = this.socketService.getProgressUpdates().subscribe((progress) => {
      this.progress = progress;
      console.log('Progress:', progress); 
      this.cdr.detectChanges(); // Ensure the UI updates with the new progress
    });
  }

  ngOnDestroy(): void {
    if (this.progressSubscription) {
      this.progressSubscription.unsubscribe();
    }
    this.socketService.socketDisconnect();
  }

  onNoClick(): void {
    this.dialogRef.close();
  }

  onYesClick(): void {
    if (this.data.onConfirm) {
      this.data.onConfirm();
    }
    this.dialogRef.close(true);
  }

  updateData(newData: any): void {
    Object.assign(this.data, newData);
    this.cdr.detectChanges();
  }
}
