// Defines the structure of the sensitive variables data that is used in the .yml files.
export const SensitiveVariables:string=`
extensions:
  - addsTo:
      pack: __PACK_NAME__
      extensible: sensitiveVariables
    data:
    - ["fileName", "sensitiveVariable"]
----------
`