import { Component, Inject, ChangeDetectorRef } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';

@Component({
  selector: 'app-gpt-progress-dialog',
  templateUrl: './gpt-progress-dialog.component.html',
})
export class GPTProgressDialogComponent {
  constructor(
    public dialogRef: MatDialogRef<GPTProgressDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: any,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    // Manually trigger change detection if needed
    this.cdr.detectChanges();
  }

  onClose(): void {
    this.dialogRef.close();
  }
}
