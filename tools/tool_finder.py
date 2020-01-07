from tools.art.adversarial_attacks import AdversarialAttacks as ArtAttacks
from tools.cleverhans.adversarial_attacks import AdversarialAttacks as CleverHansAttacks
from tools.foolbox.adversarial_attacks import AdversarialAttacks as FoolboxAttacks

TOOLS_BY_NAME = {
    'art': ArtAttacks,
    'cleverhans': CleverHansAttacks,
    'foolbox': FoolboxAttacks
}


def find_tool(tool_name):
    if tool_name.lower() in TOOLS_BY_NAME:
        return TOOLS_BY_NAME[tool_name.lower()]
    raise Exception(f"Attention: \"{tool_name}\" is not an available tool!")
