import re, csv, codecs, cStringIO
from ast import literal_eval
from wikitools.wiki import Wiki
from wikitools.api import APIRequest

API_URL = "http://localhost:8888/wiki/api.php" #"https://en.wikipedia.org/w/api.php"
MISSING = "--MISSING--"

class WikipediaPage(object):
  def __init__(self, title, summary, categories):
    self.title = title 
    self.summary = summary 
    self.categories = set(categories) 

  @staticmethod
  def fromResponse(response):
    title = response.get(u'title', MISSING) 
    summary = response.get(u'extract', MISSING)
    if summary != MISSING:
      lua = re.compile(r"\{\{(.+)\}\}", re.DOTALL)
      summary = re.sub(lua, "", summary).strip()
    if  u'categories' in response:
      categories = [category[u'title'] for category in response[u'categories']]
    else:
      categories = MISSING
    return WikipediaPage(title,summary,categories)    

  @staticmethod
  def fromList(stringList):
    title, summary, categories = stringList
    categories = literal_eval(categories) 
    return WikipediaPage(title,summary,categories)    

  def isComplete(self):
    return MISSING not in (self.title, self.summary, self.categories)

  def isDisambiguation(self):
    return u'Category:All disambiguation pages' in self.categories
  
  def __repr__(self):
    rep = u'Title: %s\n\nSummary: %s\n\nCategories: %s' % (self.title, self.summary, self.categories)
    return rep.encode('utf8')

  def toList(self):
    return [self.title, self.summary, unicode(self.categories)]

class WikipediaClient(object):
  def __init__(self):
    self.wiki = Wiki(API_URL)
    
  def randomPages(self, numPages=1):
    pageIds = []
    # Can only get 10 pages at a time
    while numPages > 0:
      limit = min(numPages, 10)
      params = {'action' : 'query',\
                'list': 'random',\
                'rnlimit': limit,\
                'rnnamespace': 0,\
                'formatversion':2}
      response = self._query(params)[u'query'][u'random']
      pageIds.extend([page[u'id'] for page in response])
      numPages -= 10
    return [self.rageById(pageId) for pageId in pageIds]

  def pageByTitle(self, title):
    params = {'titles': title}
    return self._page(params)
 
  def pageById(self, pageId):
    params = {'pageids': pageId}
    return self._page(params)

  def _page(self, overrides): 
    params = {'action' : 'query',\
              'prop': 'extracts|categories',\
              'clshow': '!hidden',\
              'explaintext': '',\
              'exintro': '',\
              'formatversion':2}
    for key, value in overrides.items():
      params[key] = value 
    response = self._query(params)[u'query'][u'pages'][0]
    return WikipediaPage.fromResponse(response)

  def _query(self, params):
    request = APIRequest(self.wiki, params)
    
    return request.query(querycontinue=False)

### FROM PYTHON API
class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")

class UnicodeReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
