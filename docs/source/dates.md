# Date Transforms

## Date transformer types
- date: Conserves date
- time: Conserves time
- datetime: Conserves both

### Description
* dates are always relative to a monday to preserve day patterns
* times are always relative to 00:00 to preserve time patterns
* Dates are first transformed into 

params:

  - **date**: 
      - year: years: 0-N, weeks 0-51, day-of-week 0-6
      - week: weeks: 0-N, day-of-week 0-6
      - day: days 0-N (0 is a monday)
      - default: year
  
  - **time**:
      - seconds: hours: 0-23, minutes: 0-59, seconds: 0-59
      - half-minutes: hours: 0-23, minutes: 0-59, half-minute: 0-1
      - minutes: hours: 0-23, minutes: 0-59
      - half-hours: hours: 0-23, half-hours: 0-1
      - hours: hours: 0-23
  
  - **ref**:
    - `<table>.<column>`
    - table is optional
    - the date to which all other dates are relative to
    - if not provided, the minimum date of that column is used as a reference

    - the reference column of a table should reference itself to get a day number
    - Example: an admission table is linked to an events table. 
      The admissions table contains the start date of an admission all
      of the other dates will be relative to. 
      When used as a reference, all other dates will be relative to the 00:00 time on Monday of that week (before the admission happened). Therefore, it is useful to keep the admission column.