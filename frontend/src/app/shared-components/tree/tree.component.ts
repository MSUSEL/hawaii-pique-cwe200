import { FlatTreeControl } from "@angular/cdk/tree";
import {
    Component,
    Directive,
    Input,
    OnChanges,
    OnInit,
    SimpleChanges,
    Output,
    EventEmitter,
} from "@angular/core";
import {
    MatTreeFlatDataSource,
    MatTreeFlattener,
} from "@angular/material/tree";
import { EditorService } from "../editor-service";

export class ItemNode {

    children: ItemNode[];
    name: string;
    type:string;
    fullPath:string;
    level:number=0;
    constructor(name:string='',type:string='',fullPath:string='',children:ItemNode[]=null){
        this.name=name;
        this.type=type;
        this.fullPath=fullPath;
        this.children=children;
    }
}
export interface Region {
    startLine: number;
    startColumn: number;
    endLine: number;
    endColumn: number;
}

export class ItemFlatNode {
    name: string;
    level: number;
    type: string;
    fullPath: string;
    expandable: boolean;
    code: string = '';
    region?: Region;  // Adding region property
}
@Directive()
export class Tree {
    constructor(protected editorService:EditorService) {}

    hasChild = (_: number, node: ItemFlatNode) => node.expandable;
    getLevel = (node: ItemFlatNode) => node.level;

    protected _transformer = (node: ItemNode, level: number) => {
        return {
            expandable: !!node.children && node.children.length > 0,
            name: node.name,
            level: level,
            type: node.type,
            fullPath:node.fullPath,
            code:''
        };
    };

    treeControl = new FlatTreeControl<ItemFlatNode>(
        (node) => node.level,
        (node) => node.expandable
    );

    treeFlattener = new MatTreeFlattener(
        this._transformer,
        (node) => node.level,
        (node) => node.expandable,
        (node) => node.children
    );

    dataSource = new MatTreeFlatDataSource(
        this.treeControl,
        this.treeFlattener
    );

    findCurrentNode(nodeName: string) {
        if (nodeName != null) {
            const Index = this.treeControl.dataNodes.findIndex(
                (n) => n.name == nodeName
            );
            const currentNode = this.treeControl.dataNodes[Index];
            if (currentNode) this.treeControl.expand(currentNode);
            else this.treeControl.expandAll();
        } else {
            this.treeControl.expandAll();
        }
    }

    getParentNode(node: ItemFlatNode): ItemFlatNode | null {
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
}



@Component({
    selector: "directory-tree",
    templateUrl: "./directory-tree.html",
    styleUrls: ["./tree.scss"],
})
export class DirectoryTreeComponent extends Tree implements OnInit {
    
    @Input() treeData: any[] = [];
    @Output() viewChange: EventEmitter<any> = new EventEmitter<any>();
    ngOnInit() {
        this.dataSource.data = this.treeData;
        //this.treeControl.expandAll();
    }

    ngOnChanges(changes: SimpleChanges) {
        if (changes['treeData'] && changes['treeData'].currentValue) {
            this.dataSource.data = this.treeData;
            //this.treeControl.expandAll();
        }
    }

    getFileContents(node:ItemFlatNode){
        this.viewChange.emit(node);
        this.editorService.findFile(node);
    }
}
