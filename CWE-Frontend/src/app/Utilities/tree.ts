
import { FlatTreeControl } from '@angular/cdk/tree';
import { Directive } from '@angular/core';
import {
    MatTreeFlatDataSource,
    MatTreeFlattener,
} from '@angular/material/tree';



interface Node {
    type:string;
    name: string;
    children?: Node[];
}

export interface FlatNode {
    expandable: boolean;
    name: string;
    level: number;
    type:string;
}


@Directive()
export class Tree{

    constructor(){}
    hasChild = (_: number, node: FlatNode) => node.expandable;
    getLevel = (node: FlatNode) => node.level;

    protected _transformer = (node: Node, level: number) => {
        return {
            expandable: !!node.children && node.children.length > 0,
            name: node.name,
            level: level,
            type:node.type,
        };
    };

    treeControl = new FlatTreeControl<FlatNode>(
        (node) => node.level,
        (node) => node.expandable
    );

    treeFlattener = new MatTreeFlattener(
        this._transformer,
        (node) => node.level,
        (node) => node.expandable,
        (node) => node.children,
    );

    dataSource = new MatTreeFlatDataSource(
        this.treeControl,
        this.treeFlattener
    );

    findCurrentNode(nodeName:string){
        if(nodeName!=null){
            const Index = this.treeControl.dataNodes.findIndex(
                (n) => n.name == nodeName
            );
            const currentNode = this.treeControl.dataNodes[Index];
            if(currentNode) this.treeControl.expand(currentNode);
            
        }
    }

    getParentNode(node: FlatNode): FlatNode | null {
        const currentLevel = this.getLevel(node);
        if (currentLevel < 1) {
            return null;
        }
        const startIndex = this.treeControl.dataNodes.indexOf(node) - 1;
        for (let i = startIndex; i >= 0; i--) {
            const currentNode = this.treeControl.dataNodes[i];
            if (this.getLevel(currentNode) < currentLevel) {
                return currentNode;
            }
        }
        return null;
    }

    getRootNode(node: FlatNode):FlatNode | null{
        const currentLevel = this.getLevel(node);
        if (currentLevel < 1) {
            return node;
        }
        const startIndex = this.treeControl.dataNodes.indexOf(node) - 1;
        for (let i = startIndex; i >= 0; i--) {
            const currentNode = this.treeControl.dataNodes[i];
            if (this.getLevel(currentNode) < currentLevel) {
                return this.getRootNode(currentNode);
            }
        }
        return null;

    }




}