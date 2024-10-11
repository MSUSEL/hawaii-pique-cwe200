import { Component, Input, OnInit } from '@angular/core';
import { DataFlowService } from 'src/app/Services/data-flow.service';
import { EditorService } from 'src/app/shared-components/editor-service';  // Import the EditorService
import { ItemFlatNode } from 'src/app/shared-components/tree/tree.component'; // Import ItemFlatNode

export interface FlowNode {
  message: string;  // The message to display
  uri: string;      // The file URI (not displayed, but kept for navigation)
  startLine: number;     // The line number (not displayed, but kept for navigation)
  startColumn: number;
  endColumn: number;   // The column number (not displayed, but kept for navigation)
  endLine: number;
}

@Component({
  selector: 'app-data-flow-tree',
  templateUrl: './data-flow-tree.component.html',
  styleUrls: ['./data-flow-tree.component.scss']
})
export class DataFlowTreeComponent implements OnInit {
  @Input() treeData: FlowNode[] = [];  // Full tree data with the FlowNode interface
  hoveredIndex: number = -1;  // Track hovered item
  isSubscribed: boolean = false;

  constructor(
    private dataFlowService: DataFlowService,
    private editorService: EditorService  // Inject the EditorService
  ) {}

  ngOnInit(): void {
    // Subscribe to the data flow change observable and update the tree
    if (!this.isSubscribed) {
      this.dataFlowService.dataFlowChangeObservable.subscribe((data) => {
        if (data) {
          this.treeData = data;  // Update tree data with the response
        }
      });
      this.isSubscribed = true;
    }
  }

  // Handle node click to navigate to the relevant file and line in the editor
  onNodeClick(node: FlowNode): void {
    console.log('Node clicked:', node);

    const pathComponents = node.uri.split(/[/\\]+/);
    const projectName = pathComponents[0];  

    let fullPath = node.uri;
    if (!fullPath.startsWith('Files/')) {
        fullPath = `Files/${projectName}/${fullPath}`;
    }

    const fileNode: ItemFlatNode = {
        name: fullPath.split(/[/\\]+/).pop(),
        fullPath: fullPath,
        level: 0,
        type: 'file',
        expandable: false,
        code: '',
        region: {
            startLine: node.startLine,
            startColumn: node.startColumn,
            endLine: node.endLine,
            endColumn: node.endColumn
        }
    };

    // Pass the callback to highlight after the file is loaded
    this.editorService.findFile(fileNode, () => {
        // Highlight logic will be handled by the CodeEditorComponent
    });
  }
}