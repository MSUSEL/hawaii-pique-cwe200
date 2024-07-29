import { Component, OnInit, OnDestroy } from '@angular/core';
import { CodeQlService } from 'src/app/Services/codeql-service';
import { FilesService } from 'src/app/Services/fileService';
import { SocketService } from 'src/app/Services/socket-service.service';
import { CVEUtilService } from 'src/app/attack-surface/cwe-util.service';
import { EditorService } from 'src/app/shared-components/editor-service';
import { MatDialog } from '@angular/material/dialog';
import { ChatGptService } from 'src/app/Services/chatgpt.service';
import { Subscription } from 'rxjs';
import { GPTProgressDialogComponent } from '../page-header/dialogs/gpt-progress-dialog.component';
import { ChangeDetectorRef } from '@angular/core';

declare var $: any;

@Component({
  selector: 'app-page-header',
  templateUrl: './page-header.component.html',
})
export class PageHeaderComponent implements OnInit, OnDestroy {
  isLoading: boolean = false;
  calculatedCost: number | null = null;
  private subscriptions: Subscription[] = [];

  constructor(
    public utilService: CVEUtilService,
    private socketService: SocketService,
    private fileService: FilesService,
    private codeQlService: CodeQlService,
    private chatGptService: ChatGptService,
    private editorService: EditorService,
    public dialog: MatDialog,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    $(document).ready(function () {
      $('#right-sidebar-collapse').on('click', function () {
        $('.wrapper').toggleClass('overflow-hidden');
        $('.right-sidebar').toggleClass('active');
        $('#right-sidebar-collapse i').toggleClass('fa-flip-horizontal');
      });
      $('#left-sidebar-collapse').on('click', function () {
        $('.left-sidebar').toggleClass('active');
        $('#left-sidebar-collapse i').toggleClass('fa-flip-horizontal');
      });

      $('#terminal-collapse,#terminal-dismiss').on('click', function () {
        $('#terminal').toggleClass('d-none');
      });
    });
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  uploadProject() {
    var data: FormData = new FormData();
    data.append('file', this.utilService.SelectedProject, this.utilService.ProjectName);
    this.isLoading = true;

    this.fileService.uploadFile(data).subscribe(
      (response) => {
        this.utilService.directoryFilesTree = response;
        this.utilService.isUploaded = true;
        this.isLoading = false;
      },
      (error) => {
        this.isLoading = false;
        console.error(error);
      }
    );
  }

  async runCodeQl() {
    this.socketService.clearOutput();
    await this.socketService.socketConnect();
    this.isLoading = true;

    const dialogRef = this.dialog.open(GPTProgressDialogComponent, {
      width: '500px',
      disableClose: true, // Make the dialog non-dismissable when clicking outside
      data: {
        parsingProgress: 0,
        sensitiveVariables: -1,
        sensitiveStrings: -1,
        sensitiveComments: -1,
        sensitiveSinks: -1,
        processingCodeQL: -1,
        buildProgress: -1,
      },
      position: {
        top: '100px'
      }
    });

    // Subscribe to parsing progress updates
    this.subscriptions.push(
      this.socketService.getParsingProgress().subscribe(progress => {
        console.log('parsingProgress update:', progress);
        dialogRef.componentInstance.data.parsingProgress = progress;
        if (progress >= 100) {
          dialogRef.componentInstance.data.sensitiveVariables = 0;
        }
        this.cdr.detectChanges();  // Manually trigger change detection
      })
    );

    // Subscribe to progress updates
    this.subscriptions.push(
      this.socketService.getGPTProgressVariables().subscribe(progress => {
        console.log('GPTProgressVariables update:', progress);
        dialogRef.componentInstance.data.sensitiveVariables = progress;
        if (progress >= 100) {
          dialogRef.componentInstance.data.sensitiveStrings = 0;
        }
        this.cdr.detectChanges();  // Manually trigger change detection
      })
    );

    this.subscriptions.push(
      this.socketService.getGPTProgressStrings().subscribe(progress => {
        console.log('GPTProgressStrings update:', progress);
        dialogRef.componentInstance.data.sensitiveStrings = progress;
        if (progress >= 100) {
          dialogRef.componentInstance.data.sensitiveComments = 0;
        }
        this.cdr.detectChanges();  // Manually trigger change detection
      })
    );

    this.subscriptions.push(
      this.socketService.getGPTProgressComments().subscribe(progress => {
        console.log('GPTProgressComments update:', progress);
        dialogRef.componentInstance.data.sensitiveComments = progress;
        if (progress >= 100) {
          dialogRef.componentInstance.data.sensitiveSinks = 0;
        }
        this.cdr.detectChanges();  // Manually trigger change detection
      })
    );

    this.subscriptions.push(
      this.socketService.getGPTProgressSinks().subscribe(progress => {
        console.log('GPTProgressSinks update:', progress);
        dialogRef.componentInstance.data.sensitiveSinks = progress;
        if (progress >= 100) {
          dialogRef.componentInstance.data.processingCodeQL = 0;
        }
        this.cdr.detectChanges();  // Manually trigger change detection
      })
    );

    this.subscriptions.push(
      this.socketService.getCodeQLProgress().subscribe(progress => {
        console.log('CodeQLProgress update:', progress);
        dialogRef.componentInstance.data.processingCodeQL = progress;
        if (progress >= 100) {
          dialogRef.componentInstance.data.buildProgress = 0;
        }
        this.cdr.detectChanges();  // Manually trigger change detection
      })
    );

    this.codeQlService.runCodeQl({ project: this.utilService.ProjectName }).subscribe(
      (response) => {
        this.editorService.rulesTree = response.rulesTree;
        this.editorService.locationsTree = response.locationsTree;
        this.isLoading = false;
        this.socketService.socketDisconnect();
        dialogRef.componentInstance.data.buildProgress = 100;
        this.unsubscribeAll();
        this.cdr.detectChanges();  // Manually trigger change detection
      },
      (error) => {
        this.isLoading = false;
        this.socketService.socketDisconnect();
        this.unsubscribeAll();
        console.error(error);
        dialogRef.close();
      }
    );
  }

  async usePrevious() {
    this.socketService.clearOutput();
    await this.socketService.socketConnect();
    this.isLoading = true;

    this.codeQlService.getSarifResult(this.utilService.ProjectName).subscribe(
      (response) => {
        try {
          this.editorService.rulesTree = response.rulesTree;
          this.editorService.locationsTree = response.locationsTree;
          this.isLoading = false;
          this.socketService.socketDisconnect();
        } catch (e) {
          this.isLoading = false;
          this.socketService.socketDisconnect();
        }
      },
      (error) => {
        this.isLoading = false;
        this.socketService.socketDisconnect();
        console.error(error);
      }
    );
  }

  private unsubscribeAll() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }
}
