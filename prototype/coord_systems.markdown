##### Coordinate systems

There is few coordinate systems for text analyze used in project

1. **position:int**
  - global character index in string with all text after preprocessing
  - *usage*: 
    - sampling (sequencing)
    - alignment
    - cluster detection

2. **page coordinates:tuple(int,int)** 
  - page index (for file access) and line index on page
  - *usages*: 
    - in graph engine for coloration drawing
    - for metrics scoring in predicted areas

3. **real page coordinates:tuple(int,int)**
  - `real` page number and line number in page
  - page number taken from OCR without upper page number digits
  - *usages*: 
    - for document marking by accessors
    - for metrics scoring in `true` areas
