# 5\_wikipedia Dataset

## Used for CS221 Project: Paragraph Topic Extraction
  Project Team: Ed Ng, Eugene Nho, Sigtryggur Kjartansson

### Source 
 This data is scraped from Wikipedia articles from the enwiki-latest-pages-articles.xml on Nov 14 2016
 
### Content
 We chose 5 Wikipedia categories "computer science", "mathematics", "music", "film", and "politics" and
 traversed each category hierarchy to depth 5, scraping all article summaries at each level.
 The resulting data set has input text and target list of 

### Data Stats
  Total Number of Samples: 394059
  Training set: 354419 (90%)
  Test set: 39640 (10%)
  4.1% overlapping categories, and nearly all of them were 2 categories. 
  
  Fun Fact: This [Indian Film](https://en.wikipedia.org/wiki/Jai_Bajarangabhali) was the only article
  that belonged to all 5 categories, but that's likely due to miscategorization.

