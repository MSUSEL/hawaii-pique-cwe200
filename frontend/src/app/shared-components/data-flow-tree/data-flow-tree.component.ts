import { Component, Input, OnInit } from '@angular/core';
import { DataFlowService } from 'src/app/Services/data-flow.service';
import { EditorService } from 'src/app/shared-components/editor-service';  // Import the EditorService
import { ItemFlatNode } from 'src/app/shared-components/tree/tree.component'; // Import ItemFlatNode

export interface FlowNode {
  message: string;  // The message to display
  uri: string;      // The file URI (not displayed, but kept for navigation)
  Line: number;     // The line number (not displayed, but kept for navigation)
  Column: number;   // The column number (not displayed, but kept for navigation)
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
    // console.log('Node clicked:', node);

    // Construct an ItemFlatNode with the necessary properties
    const fileNode: ItemFlatNode = {
      name: node.uri.split(/[/\\]+/).pop(),      
      fullPath: node.uri,
      level: 0,  // Set the appropriate level if needed
      type: 'file',  // Set the type (you can adjust this if needed)
      expandable: false,  // Files aren't expandable
      code: '',  // Empty code as it's loaded dynamically
      region: {  // Set the region for scrolling
        startLine: node.Line,
        startColumn: node.Column,
        endLine: node.Line,
        endColumn: node.Column
      }
    };


    // Call findFile from the editor service to open the file at the correct line and column
    this.editorService.findFile(fileNode);
  }
}
