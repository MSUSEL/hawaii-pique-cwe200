export const Sinks:string=`
extensions:
  - addsTo:
      pack: custom-codeql-queries
      extensible: sinks
    data:
    - ["fileName", "sinkName", "type"]
----------
`