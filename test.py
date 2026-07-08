import json

with open("kb.json", encoding="utf-8") as f:
    kb = json.load(f)

from pprint import pprint
import random

eid = random.choice(list(kb["entities"].keys()))
pprint(kb["entities"][eid])