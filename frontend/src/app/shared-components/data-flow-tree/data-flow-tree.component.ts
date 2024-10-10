import { FlatTreeControl } from '@angular/cdk/tree';
import { Component, Input, OnInit } from '@angular/core';
import { MatTreeFlatDataSource, MatTreeFlattener } from '@angular/material/tree';
import { Output, EventEmitter } from '@angular/core';
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

    constructor(private codeQlService: CodeQlService) {
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
        this.nodeClicked.emit(node.name); // Emit the node name when clicked
    }

    getResults(node: any) {
        // Call the CodeQL Service to fetch the data flow for the selected node
        const project = 'your_project_name'; // Add your project name here or pass it from the component using this
        this.codeQlService.getVulnerabilityTree(node.fullPath, project)
            .subscribe((response: FlowNode[]) => {
                this.treeData = response;  // Update the treeData with the response
                this.dataSource.data = this.treeData;  // Update the data source
            }, error => {
                console.error('Error fetching data flow tree:', error);
            });
    }
}
