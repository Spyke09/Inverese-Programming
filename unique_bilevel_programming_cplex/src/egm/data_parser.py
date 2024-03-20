import json
import logging
import typing as tp
from dataclasses import dataclass
from datetime import datetime

from unique_bilevel_programming_cplex.src.base.common import LPNan, LPFloat


@dataclass
class EGMData:
    cc_list_full: tp.Any
    consumption_production_assoc: tp.Any
    export_assoc: tp.Any
    graph_db: tp.Any
    prices_assoc: tp.Any
    storage_db: tp.Any
    terminal_db: tp.Any


class DataParser:
    _logger = logging.getLogger("DataParser")

    @staticmethod
    def _process_num(num):
        return LPNan if num == "Missing" else LPFloat(num)

    @staticmethod
    def _process_date(date):
        return date if isinstance(date, datetime) else datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')

    @staticmethod
    def get_data():
        DataParser._logger.info("Starting to read and pre-process data.")
        with open("../../data/ccListFull.json", "r") as f:
            cc_list_full = set(json.load(f))
        with open("../../data/consumptionProductionAssoc.json", "r") as f:
            consumption_production_assoc = json.load(f)
            consumption_production_assoc['consumption'] = consumption_production_assoc['consumption']["bcm"]
            consumption_production_assoc['production'] = consumption_production_assoc['production']["bcm"]
            consumption_production_assoc = {
                name: {
                    cou: {DataParser._process_date(d): DataParser._process_num(c) for d, c in dtc.items()}
                    for cou, dtc in pc.items()
                }
                for name, pc in consumption_production_assoc.items()
            }
        with open("../../data/exportAssoc.json", "r") as f:
            export_assoc = json.load(f)
            export_assoc = export_assoc["bcm"]
            export_assoc = {
                c1: {
                    c2: {
                        DataParser._process_date(d): DataParser._process_num(c) for d, c in expo.items()
                    }
                    for c2, expo in assoc.items()
                }
                for c1, assoc in export_assoc.items()
            }
        with open("../../data/graphDB.json", "r") as f:
            graph_db = json.load(f)
            graph_db['arcCapTimeAssoc'] = {
                DataParser._process_date(d): {(edge[0], edge[1]): DataParser._process_num(edge[2]) for edge in edges}
                for d, edges in graph_db['arcCapTimeAssoc'].items()
            }
            graph_db['arcList'] = set(tuple(i) for i in graph_db['arcList'])
            graph_db['tsoList'] = set(graph_db['tsoList'])
            graph_db['lngList'] = set(graph_db['lngList'])
            graph_db['storList'] = set(graph_db['storList'])
            graph_db['consumVertexList'] = set(graph_db['consumVertexList'])
            graph_db['consumList'] = set(graph_db['consumList'])
            graph_db['prodVertexList'] = set(graph_db['prodVertexList'])
            graph_db['prodList'] = set(graph_db['prodList'])
            graph_db['exporterVertexList'] = set(graph_db['exporterVertexList'])
            graph_db['exporterList'] = set(graph_db['exporterList'])
            graph_db['exportDirections'] = {i: set(j) for i, j in graph_db['exportDirections'].items()}
        with open("../../data/priceAssoc.json", "r") as f:
            prices_assoc = json.load(f)
            prices_assoc = {
                name: {
                    DataParser._process_date(d): DataParser._process_num(n) for d, n in pc.items()
                }
                for name, pc in prices_assoc.items()
            }
        with open("../../data/storageDB.json", "r") as f:
            storage_db = json.load(f)
            storage_db = storage_db["aggregated"]
            storage_db = {
                name: {
                    "CC": st["CC"],
                    "DayData": {DataParser._process_date(d): {c: DataParser._process_num(n) for c, n in ns.items()}
                                for d, ns in st["DayData"].items()},
                    "MonthData": {DataParser._process_date(d): {c: DataParser._process_num(n) for c, n in ns.items()}
                                  for d, ns in st["MonthData"].items()}
                }
                for name, st in storage_db.items()
            }
        with open("../../data/terminalDB.json", "r") as f:
            terminal_db = json.load(f)
            terminal_db = {
                name: {
                    "CC": st["CC"],
                    "DayData": {DataParser._process_date(d): {c: DataParser._process_num(n) for c, n in ns.items()}
                                for d, ns in st["DayData"].items()},
                    "MonthData": {DataParser._process_date(d): {c: DataParser._process_num(n) for c, n in ns.items()}
                                  for d, ns in st["MonthData"].items()}
                }
                for name, st in terminal_db.items()
            }

        DataParser._logger.info("Reading and preprocessing data is finished.")
        return EGMData(
            cc_list_full,
            consumption_production_assoc,
            export_assoc,
            graph_db,
            prices_assoc,
            storage_db,
            terminal_db
        )
