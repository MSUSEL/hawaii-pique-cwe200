// Defines the structure of the sinks data that is used in the .yml files.
export const Sinks:string=`
extensions:
  - addsTo:
      pack: custom-codeql-queries
      extensible: sinks
    data:
    - ["fileName", "sinkName", "type"]
----------
`