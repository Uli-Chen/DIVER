"""Method registry; implementations live in each named baseline directory."""
from baselines.TGTB.attack import Attack as TgtbAttack, prompt as tgtb_prompt
from baselines.PIDE.attack import Attack as PideAttack, prompt as pide_prompt
from baselines.IKEA.attack import IkeaAttack, ikea_definitions

TOPICS = {
    'novel': 'fictional novels and stories, their characters, locations, objects, events and relationships',
    'medical': 'general medical knowledge, diseases, symptoms, diagnosis, treatments, drugs and their relationships',
    'agriculture': 'agriculture, crops, soil, irrigation, fertilizers, pests, farming practices and their relationships',
}
TOPIC = TOPICS['novel']

def baseline_prompt(method, anchor):
    return {'TGTB': tgtb_prompt, 'PIDE': pide_prompt}[method](anchor)

def StaticAttack(method, seed, anchors):
    return {'TGTB': TgtbAttack, 'PIDE': PideAttack}[method](seed, anchors)
