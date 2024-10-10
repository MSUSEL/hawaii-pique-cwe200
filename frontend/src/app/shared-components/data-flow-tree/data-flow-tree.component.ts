import { Component, Input, OnInit } from '@angular/core';
import { DataFlowService } from 'src/app/Services/data-flow.service'; 

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

  constructor(private dataFlowService: DataFlowService) {}

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

  // Handle node click to eventually navigate to the relevant file and line
  onNodeClick(node: FlowNode): void {
    console.log('Node clicked:', node);
    this.navigateToCode(node.uri, node.Line, node.Column);
  }

  navigateToCode(uri: string, line: number, column: number): void {
    console.log(`Navigating to file: ${uri}, Line: ${line}, Column: ${column}`);
    // Logic to navigate to file and line in code editor (e.g., using a service)
  }
}
