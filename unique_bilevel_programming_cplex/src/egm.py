import json
from dataclasses import dataclass
import typing as tp


@dataclass
class EGMData:
    cc_list_full: tp.Any
    consumption_production_assoc: tp.Any
    export_assoc: tp.Any
    graph_db: tp.Any
    prices_assoc: tp.Any
    storage_db: tp.Any
    terminal_db: tp.Any


def get_data():
    with open("../data/ccListFull.json", "r") as f:
        cc_list_full = json.load(f)
    with open("../data/consumptionProductionAssoc.json", "r") as f:
        consumption_production_assoc = json.load(f)
        consumption_production_assoc['consumption'] = consumption_production_assoc['consumption']["bcm"]
        consumption_production_assoc['production'] = consumption_production_assoc['production']["bcm"]
    with open("../data/exportAssoc.json", "r") as f:
        export_assoc = json.load(f)
        export_assoc = export_assoc["bcm"]
    with open("../data/graphDB.json", "r") as f:
        graph_db = json.load(f)
    with open("../data/priceAssoc.json", "r") as f:
        prices_assoc = json.load(f)
    with open("../data/storageDB.json", "r") as f:
        storage_db = json.load(f)
        storage_db = storage_db["aggregated"]
    with open("../data/terminalDB.json", "r") as f:
        terminal_db = json.load(f)

    return EGMData(
        cc_list_full,
        consumption_production_assoc,
        export_assoc,
        graph_db,
        prices_assoc,
        storage_db,
        terminal_db
    )

get_data()
