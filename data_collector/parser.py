import os
import csv
import threading
from tools import WikipediaClient, UnicodeReader, UnicodeWriter

dataPath = os.path.join(os.path.dirname(os.path.realpath(__file__)) ,"data")

topics = ["computer_science", "mathematics", "music", "film", "politics"]

pageIdToTopics = {}

for topic in topics:
  with open(os.path.join(dataPath, topic + ".csv"), "rb") as topicFile:
    reader = UnicodeReader(topicFile, escapechar= "\\")
    for line in reader:
      pageId = line[2]
      if not pageId:
        print line
      pageIdToTopics[pageId] = pageIdToTopics.get(pageId, []) + [topic]  


class SummaryGetter(threading.Thread):
  def __init__(self, threadID, writer, pageIDsAndTopics):
    threading.Thread.__init__(self)
    self.threadID = threadID
    self.writer = writer
    self.pageIDsAndTopics = pageIDsAndTopics
 
  def run(self):
    rows = []
    client = WikipediaClient()
    for pageID, topics in self.pageIDsAndTopics:
      summary = client.pageById(pageID).summary
      rows.append([summary, unicode(topics)])
    self.writerows(rows)
 
  def writerows(self, rows):
    with writeLock:
      self.writer.writerows(rows)    

writeLock = threading.Lock()    

with open(os.path.join(dataPath, "wikipedia-data.csv"), "wb") as dataFile:
  writer = UnicodeWriter(dataFile)
  client = WikipediaClient()
  pageIdAndTopics = pageIdToTopics.items()
  numPages = len(pageIdAndTopics)
  numThreads = 20
  chunkSize = 100
  chunks = [pageIdAndTopics[i:i + chunkSize] for i in xrange(0, numPages, chunkSize)]
  for chunkIndex in xrange((numPages / (numThreads * chunkSize)) + 1):
    chunkStart = chunkIndex * numThreads 
    threads = []
    for i, chunk in enumerate(chunks[chunkStart:chunkStart + numThreads]):
      threads.append(SummaryGetter(i, writer, chunk))

    for thread in threads:
      thread.start()

    for thread in threads:
      thread.join() 
