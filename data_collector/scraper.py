from tools import WikipediaClient, UnicodeWriter, UnicodeReader
import sys, csv, datetime, threading

class WikipediaScraper(threading.Thread):
  def __init__(self, threadID, writer, numPages):
    threading.Thread.__init__(self)
    self.threadID = threadID
    self.writer = writer
    self.numPages = numPages

  def run(self):
    client = WikipediaClient()
    pages = client.randomPages(numPages=self.numPages)
    self.write(pages)
     
  def write(self, pages):
    with writeLock:
      for page in pages:
        if page.isComplete() and not page.isDisambiguation():
          # Don't write incomplete or disambiguation pages
          self.writer.writerow(page.toList())
       
if __name__ == '__main__':
  writeLock = threading.Lock()

  csvFileName = "data/wikipedia-{}.csv".format(datetime.datetime.now())
  with open(csvFileName, 'wb') as csvFile:
    writer = UnicodeWriter(csvFile)
    threads = []
    for i in range(100):
      threads.append(WikipediaScraper(i, writer, 40))
  
    for thread in threads:
      thread.start()
   
    for thread in threads:
      thread.join()
  
  print "Wikipedia Data File %s created" % csvFileName

  with open(csvFileName, 'rb') as csvFile:
    reader = UnicodeReader(csvFile)
    count = 0
    for row in reader:
      count +=1
  print "Successfully wrote %d rows" % count
