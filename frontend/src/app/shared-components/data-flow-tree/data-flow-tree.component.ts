import { FlatTreeControl } from '@angular/cdk/tree';
import { Component, Input, OnInit } from '@angular/core';
import { MatTreeFlatDataSource, MatTreeFlattener } from '@angular/material/tree';
import { Output, EventEmitter } from '@angular/core';  // Import these
import { CodeQlService } from 'src/app/Services/codeql-service';


interface FlowNode {
  name: string;
  children?: FlowNode[];
}

interface FlatFlowNode {
  expandable: boolean;
  name: string;
  level: number;
}

@Component({ 
  selector: 'app-data-flow-tree',
  templateUrl: './data-flow-tree.component.html',
  styleUrls: ['./data-flow-tree.component.scss']
})
export class DataFlowTreeComponent implements OnInit {
    @Input() treeData: FlowNode[] = []; // Receive the tree data as input
    @Output() nodeClicked = new EventEmitter<string>();
  
    treeControl: FlatTreeControl<FlatFlowNode>;
    treeFlattener: MatTreeFlattener<FlowNode, FlatFlowNode>;
    dataSource: MatTreeFlatDataSource<FlowNode, FlatFlowNode>;
  
    constructor() {
      this.treeFlattener = new MatTreeFlattener(
        this.transformer,
        (node) => node.level,
        (node) => node.expandable,
        (node) => node.children
      );
  
      this.treeControl = new FlatTreeControl<FlatFlowNode>(
        (node) => node.level,
        (node) => node.expandable
      );
  
      this.dataSource = new MatTreeFlatDataSource(this.treeControl, this.treeFlattener);
    }
  
    ngOnInit(): void {
      // Update the data source whenever treeData changes
      this.dataSource.data = this.treeData;
    }
  
    transformer = (node: FlowNode, level: number) => {
      return {
        expandable: !!node.children && node.children.length > 0,
        name: node.name,
        level: level
      };
    };
  
    hasChild = (_: number, node: FlatFlowNode) => node.expandable;

    onNodeClick(node: FlowNode) {
        // if (node.filePath) {
        //     this.nodeClicked.emit(node.filePath); // Emit the filePath when clicked
        // }
    }
    getResults(node: FlowNode) {
    
    }
  }
  