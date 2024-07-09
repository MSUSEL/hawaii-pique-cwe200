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
  progressEstimate: number = 0;
  private progressSubscription: Subscription;
  private progressEstimateSubscription: Subscription;
  isProgressComplete: boolean = false; // Flag to track completion of the first progress bar
  isEstimateComplete: boolean = false; // Flag to track completion of the last progress bar
  public dialogRef: MatDialogRef<ConfirmationDialogComponent>; // Dialog reference

  constructor(
    public dialogReference: MatDialogRef<ConfirmationDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: any,
    private socketService: SocketService,
    private cdr: ChangeDetectorRef
  ) {
    this.dialogRef = dialogReference;
  }

  ngOnInit(): void {
    this.socketService.socketConnect();
    this.progressSubscription = this.socketService.getProgressUpdates().subscribe((progress) => {
      this.progress = progress;
      console.log('Progress:', progress);
      if (progress >= 100) {
        this.isProgressComplete = true;
      }
      this.cdr.detectChanges(); // Ensure the UI updates with the new progress
      this.dialogRef.updateSize('500px','auto'); // Update the dialog size
    });

    this.progressEstimateSubscription = this.socketService.getProgressEstimate().subscribe((progressEstimate) => {
      if (this.isProgressComplete) {
        this.progressEstimate = progressEstimate;
        console.log('Progress Estimate:', progressEstimate);
        if (progressEstimate >= 100) {
          this.isEstimateComplete = true;
        }
        this.cdr.detectChanges(); // Ensure the UI updates with the new progress estimate
        this.dialogRef.updateSize('500px','auto'); // Update the dialog size
      }
    });
  }

  ngOnDestroy(): void {
    if (this.progressSubscription) {
      this.progressSubscription.unsubscribe();
    }
    if (this.progressEstimateSubscription) {
      this.progressEstimateSubscription.unsubscribe();
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
    this.dialogRef.updateSize('500px','auto'); // Update the dialog size after updating data
  }
}
