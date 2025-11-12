from acl_anthology import Anthology
import pandas as pd
from typing import List, Dict, Set
from collections import defaultdict
import re
from datetime import datetime

MIN_YEAR = 2010
def main():
    volumeId = "2020.acl-main"

    anthology = Anthology.from_repo()

    volume = anthology.get_volume(volumeId)

    print(volume.year)

if __name__ == "__main__":
    main()
    