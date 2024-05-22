import { ChangeDetectorRef, Component, Inject } from '@angular/core';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';

@Component({
  selector: 'app-confirmation-dialog',
  templateUrl: './confirm-chatgpt-cost-dialog.component.html',
})
export class ConfirmationDialogComponent {
  constructor(
    public dialogRef: MatDialogRef<ConfirmationDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: any,
    private cdr: ChangeDetectorRef
  ) {}

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
