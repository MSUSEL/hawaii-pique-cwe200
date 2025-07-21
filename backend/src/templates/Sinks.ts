// Defines the structure of the sinks data that is used in the .yml files.
export const Sinks:string=`
extensions:
  - addsTo:
      pack: __PACK_NAME__
      extensible: sinks
    data:
    - ["fileName", "sinkName", "type"]
----------
`