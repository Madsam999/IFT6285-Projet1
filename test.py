from acl_anthology import Anthology as ant
from scholarly import scholarly as sch
import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())


def main():

    anthology = ant.from_repo()

    paperId = "P16-1004"
    paper = anthology.get(paperId)
    
    paperName = str(paper.title)
    print(paperName)

    search = sch.search_pubs(paperName)
    print(next(search)["num_citations"])

    print("Hello world")

if __name__ == "__main__":
    main()