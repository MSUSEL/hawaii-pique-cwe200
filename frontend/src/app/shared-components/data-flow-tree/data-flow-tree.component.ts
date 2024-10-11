import { Component, Input, OnInit } from '@angular/core';
import { DataFlowService } from 'src/app/Services/data-flow.service';
import { EditorService } from 'src/app/shared-components/editor-service';  // Import the EditorService
import { ItemFlatNode } from 'src/app/shared-components/tree/tree.component'; // Import ItemFlatNode

export interface FlowNode {
  message: string;  // The message to display
  uri: string;      // The file URI (not displayed, but kept for navigation)
  startLine: number; // The starting line number for the highlight
  startColumn: number; // The starting column number for the highlight
  endLine: number; // The ending line number for the highlight
  endColumn: number; // The ending column number for the highlight
  isExpanded?: boolean; // To track the expansion state of the node
  type: string; // The type of the node
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
  activeTabIndex: number = -1;  // Track the last clicked tab

  constructor(
    private dataFlowService: DataFlowService,
    private editorService: EditorService  // Inject the EditorService
  ) {}

  ngOnInit(): void {
    // Subscribe to the data flow change observable and update the tree
    if (!this.isSubscribed) {
      this.dataFlowService.dataFlowChangeObservable.subscribe((data) => {
        if (data) {
          this.treeData = data.map(node => ({ ...node, isExpanded: false }));  // Initialize all nodes as collapsed
        }
      });
      this.isSubscribed = true;
    }
  }

  // Toggle node expansion and trigger highlight in editor
  onNodeClick(node: FlowNode, index: number): void {
    node.isExpanded = !node.isExpanded;  // Toggle expansion

    // Set active tab index to apply highlighting
    this.activeTabIndex = index;

    const fullPath = this.correctPath(node.uri);

    console.log('Data Flow Tree', node);  // Log the clicked node for debugging

    const fileNode: ItemFlatNode = {
        name: fullPath.split('/').pop(),
        fullPath: fullPath,
        level: 0,
        type: node.type,
        expandable: false,
        code: '',
        region: {
            startLine: node.startLine,
            startColumn: node.startColumn,
            endLine: node.endLine,
            endColumn: node.endColumn
        }
    };

    // Find and highlight the file in the editor
    this.editorService.findFile(fileNode, () => {
        // The actual highlight is handled in the CodeEditorComponent
    });
  }

  // Helper to correct file path format
  correctPath(uri: string): string {
    const pathComponents = uri.split(/[/\\]+/);
    const projectName = pathComponents[0];
    let fullPath = uri;
    if (!fullPath.startsWith('Files/')) {
        fullPath = `Files/${projectName}/${fullPath}`;
    }
    return fullPath;
  }
}
