"""Category alias tables used when scoring the predicted object name.

Two vocabularies:

* **coco** - the 80 COCO categories with hand-curated aliases, kept inline
  because they are small and worth reading in a diff. ``LOOSE_ALIASES_RISKY``
  holds broader folds (``chair`` <- ``seat``, ``car`` <- ``vehicle``) that are
  off by default; the reported numbers use the safe set only.
* **lvis** - a 3.6k-entry map generated from the official LVIS ``synonyms``
  fields, shipped as ``aliases/lvis_synonym_to_canon_safe.json``.

Both end up as the same structure: normalised alias -> canonical name.
"""

import json
from pathlib import Path

from .labels import normalize_label

ALIAS_DIR = Path(__file__).resolve().parent.parent / "aliases"

# Common COCO-ish synonym folds → canonical COCO names.
# Extend as you see new variants in your outputs.
LOOSE_ALIASES_SAFE = {
    "person": ["man","woman","boy","girl","kid","child","baby","guy","lady","people","human"],
    "bicycle": ["bike","pushbike","cycle","bicycl"],
    "car": ["auto","automobile","sedan","coupe","hatchback","taxi","cab"],
    "motorcycle": ["motorbike","motor cycle","moped","scooter"],
    "airplane": ["plane","jet","aircraft","airliner","aeroplane"],
    "bus": ["coach","school bus","minibus","city bus"],
    "train": ["locomotive","railcar","tram","subway train"],
    "truck": ["lorry","pickup","pickup truck","semi","tractor trailer","box truck","dump truck"],
    "boat": ["ship","sailboat","yacht","canoe","kayak","ferry","rowboat"],
    "traffic light": ["stoplight","traffic signal","stop light"],
    "fire hydrant": ["hydrant"],
    "stop sign": ["stop-sign","stop sign board","octagonal stop"],
    "parking meter": ["meter","pay meter","parking machine"],
    "bench": ["park bench","seat bench"],
    "bird": ["sparrow","pigeon","duck","goose","owl","eagle","hawk","seagull","parrot","swan"],
    "cat": ["kitten","kitty"],
    "dog": ["puppy","hound","bulldog","poodle","retriever","shepherd"],
    "horse": ["pony","stallion","mare"],
    "sheep": ["lamb","ewe","ram"],
    "cow": ["calf","bull","cattle"],
    "elephant": ["elephants"],
    "bear": ["grizzly","black bear","brown bear","polar bear","panda bear","panda"],
    "zebra": ["zebras"],
    "giraffe": ["giraffes"],
    "backpack": ["rucksack","knapsack","bookbag","schoolbag","daypack"],
    "umbrella": ["brolly","sunshade","parasol"],
    "handbag": ["purse","shoulder bag","satchel","hand bag","crossbody"],
    "tie": ["necktie","neck tie"],
    "suitcase": ["luggage","trolley","carry-on","valise"],
    "frisbee": ["flying disc","flying saucer toy","disc"],
    "skis": ["ski","ski gear","ski equipment"],
    "snowboard": ["snow board"],
    "sports ball": ["soccer ball","football","basketball","baseball","tennis ball","volleyball","golf ball","rugby ball"],
    "kite": ["kites"],
    "baseball bat": ["bat"],
    "baseball glove": ["mitt","baseball mitt","glove"],
    "skateboard": ["skate board","skate"],
    "surfboard": ["surf board"],
    "tennis racket": ["tennis racquet","racquet","racket"],
    "bottle": ["bottle plastic","glass bottle","water bottle"],
    "wine glass": ["wineglass","goblet"],
    "cup": ["mug","coffee cup","teacup","paper cup"],
    "fork": ["dinner fork","table fork"],
    "knife": ["table knife","steak knife","butter knife"],
    "spoon": ["teaspoon","tablespoon"],
    "bowl": ["salad bowl","soup bowl","cereal bowl"],
    "banana": ["bananas"],
    "apple": ["apples"],
    "sandwich": ["sandwiches","sub","burger sandwich"],
    "orange": ["oranges","tangerine","clementine"],
    "broccoli": ["broccol"],
    "carrot": ["carrots"],
    "hot dog": ["hotdog","frank","frankfurter"],
    "pizza": ["slice of pizza","pizza slice","pie pizza"],
    "donut": ["doughnut","ring donut","glazed donut"],
    "cake": ["cupcake","birthday cake","layer cake"],
    "chair": ["armchair","stool","seat","desk chair","dining chair"],
    "couch": ["sofa","settee","loveseat","sectional"],
    "potted plant": ["houseplant","potted flower","indoor plant","plant pot","potted tree"],
    "bed": ["beds","bunk bed","crib","cot"],
    "dining table": ["dining-table","dinner table","kitchen table"],
    "toilet": ["wc","lavatory","commode","toilet bowl","loo","john"],
    "tv": ["television","tv set","flat screen tv","smart tv"],
    "laptop": ["notebook","notebook computer","macbook","chromebook"],
    "mouse": ["computer mouse"],
    "remote": ["remote control","tv remote","clicker"],
    "keyboard": ["computer keyboard","kbd"],
    "cell phone": ["cellphone","mobile phone","mobile","smartphone","iphone","android phone","cell phone"],
    "microwave": ["microwave oven"],
    "oven": ["stove oven","electric oven","gas oven","range oven"],
    "toaster": ["bread toaster","toast maker"],
    "sink": ["kitchen sink","bathroom sink","washbasin","basin"],
    "refrigerator": ["fridge","refrigerator fridge","icebox"],
    "book": ["books","novel","paperback","hardcover"],
    "clock": ["wall clock","alarm clock","desk clock"],
    "vase": ["flower vase"],
    "scissors": ["shears"],
    "teddy bear": ["teddy","stuffed bear","plush bear","teddy toy"],
    "hair drier": ["hair dryer","hairdryer","blow dryer","blow-dryer"],
    "toothbrush": ["tooth brush"],
}

# Ambiguous/broad variants. Use only if you’re OK with a looser eval.
LOOSE_ALIASES_RISKY = {
    "tv": ["monitor","screen","flat screen","display"],
    "dining table": ["table","desk"],          # 'desk' is quite risky
    "potted plant": ["plant","greenery","shrub"],
    "handbag": ["bag","hand bag","carry bag"],
    "sports ball": ["ball"],
    "car": ["vehicle","ride"],                 # very broad
    "truck": ["vehicle"],
    "train": ["metro","subway"],               # context-dependent
    "boat": ["vessel"],
    "bench": ["bench seat","bleacher"],
    "refrigerator": ["freezer"],               # not always correct
    "microwave": ["oven microwave"],           # can confuse with 'oven'
    "chair": ["seat"],                         # broad
}


def build_coco_synonym_map(include_risky: bool = False) -> dict:
    """Build the normalised-alias -> canonical-COCO-label mapping."""
    m = {}
    for canon, aliases in LOOSE_ALIASES_SAFE.items():
        m[normalize_label(canon)] = canon  # canonical maps to itself
        for a in aliases:
            m[normalize_label(a)] = canon
    if include_risky:
        for canon, aliases in LOOSE_ALIASES_RISKY.items():
            for a in aliases:
                m[normalize_label(a)] = canon
    return m


def load_lvis_synonym_map() -> dict:
    with open(ALIAS_DIR / "lvis_synonym_to_canon_safe.json", encoding="utf-8") as f:
        return json.load(f)


def load_synonym_map(vocab: str, include_risky: bool = False) -> dict:
    """``vocab`` is ``coco`` (in-domain) or ``lvis`` (out-of-domain).

    ``include_risky`` only applies to COCO; the LVIS map ships safe-only.
    """
    if vocab == "coco":
        return build_coco_synonym_map(include_risky=include_risky)
    if vocab == "lvis":
        if include_risky:
            raise ValueError("no risky alias set is shipped for the LVIS vocabulary")
        return load_lvis_synonym_map()
    raise ValueError(f"unknown vocabulary: {vocab!r}")
