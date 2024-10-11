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
  @Input() treeData: FlowNode[][] = [];  // Now an array of arrays for separate flows
  activeFlowIndex: number = -1;  // Track the last clicked flow tab
  activeNodeIndex: number = -1;  // Track the last clicked node within a flow

  constructor(
    private dataFlowService: DataFlowService,
    private editorService: EditorService  // Inject the EditorService
  ) {}

  ngOnInit(): void {
    // Subscribe to the data flow change observable and update the tree
    this.dataFlowService.dataFlowChangeObservable.subscribe((data) => {
      if (data) {
        this.treeData = data.map(flow => flow.map(node => ({ ...node, isExpanded: false })));  // Initialize all nodes as collapsed
        this.activeFlowIndex = -1; // Reset the active flow when a new vulnerability is clicked
        this.activeNodeIndex = -1; // Reset the active node when a new vulnerability is clicked
      }
    });
  }

  // Toggle flow tab expansion and allow collapsing
  toggleFlow(flowIndex: number): void {
    if (this.activeFlowIndex === flowIndex) {
      this.activeFlowIndex = -1;  // Collapse if already expanded
    } else {
      this.activeFlowIndex = flowIndex;  // Expand the selected flow
      this.activeNodeIndex = -1;  // Reset node highlight when switching flows
    }
  }

  // Toggle node expansion and trigger highlight in editor
  onNodeClick(flowIndex: number, nodeIndex: number): void {
    // Collapse previously expanded nodes if needed
    if (this.activeNodeIndex !== -1 && this.activeFlowIndex === flowIndex) {
      this.treeData[flowIndex][this.activeNodeIndex].isExpanded = false;
    }

    const node = this.treeData[flowIndex][nodeIndex];
    node.isExpanded = !node.isExpanded;  // Toggle expansion

    // Set active node index to apply highlighting
    this.activeNodeIndex = nodeIndex;

    const fullPath = this.correctPath(node.uri);

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
