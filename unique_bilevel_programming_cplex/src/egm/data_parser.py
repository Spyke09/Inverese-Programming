import json
import logging
import typing as tp
from dataclasses import dataclass
from datetime import datetime

from dateutil import relativedelta

from unique_bilevel_programming_cplex.src.base.common import LPNan, LPFloat


@dataclass
class EGMData:
    cc_list_full: tp.Any
    cp_assoc: tp.Any
    export_assoc: tp.Any
    graph_db: tp.Any
    prices_assoc: tp.Any
    storage_db: tp.Any
    terminal_db: tp.Any


class DataParser:
    def __init__(self, dates):
        self._logger = logging.getLogger("DataParser")
        self._data = None
        self._dates = dates

        self._delta = relativedelta.relativedelta(months=1)
        self._date_from_d = self._dates[0] - self._delta
        self._date_to_d = self._dates[-1] + self._delta

    @staticmethod
    def _process_num(num, c=1e4, c_ns=1e1):
        return LPNan if num == "Missing" else LPFloat(num) * c * c_ns

    @staticmethod
    def _process_date(date):
        return date if isinstance(date, datetime) else datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')

    def _check_date(self, date, with_delta=False):
        if isinstance(date, str):
            date = self._process_date(date)
        if not with_delta:
            return self._dates[0] <= date <= self._dates[-1]
        else:
            return self._date_from_d <= date <= self._date_to_d

    def get_data(self):
        c_ns = 0.000097158
        self._logger.info("Starting to read and pre-process data.")
        with open("../../data/ccListFull.json", "r") as f:
            cc_list_full = set(json.load(f))
        with open("../../data/consumptionProductionAssoc.json", "r") as f:
            consumption_production_assoc = json.load(f)
            consumption_production_assoc['consumption'] = consumption_production_assoc['consumption']["bcm"]
            consumption_production_assoc['production'] = consumption_production_assoc['production']["bcm"]
            consumption_production_assoc = {
                name: {
                    cou: {DataParser._process_date(d): DataParser._process_num(c) for d, c in dtc.items() if self._check_date(d)}
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
                        DataParser._process_date(d): DataParser._process_num(c) for d, c in expo.items() if self._check_date(d)
                    }
                    for c2, expo in assoc.items()
                }
                for c1, assoc in export_assoc.items()
            }
        with open("../../data/graphDB.json", "r") as f:
            graph_db = json.load(f)
            graph_db['arcCapTimeAssoc'] = {
                DataParser._process_date(d):
                    {(edge[0], edge[1]): DataParser._process_num(edge[2], c_ns=c_ns) for edge in edges}
                for d, edges in graph_db['arcCapTimeAssoc'].items() if self._check_date(d)
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
                    DataParser._process_date(d): DataParser._process_num(n, 1e-3) for d, n in pc.items()
                }
                for name, pc in prices_assoc.items()
            }
        with open("../../data/storageDB.json", "r") as f:
            storage_db = json.load(f)
            storage_db = storage_db["aggregated"]
            storage_db = {
                name: {
                    "CC": st["CC"],
                    "MonthData": {
                        DataParser._process_date(d): {c: DataParser._process_num(n, c_ns=c_ns) for c, n in ns.items()}
                        for d, ns in st["MonthData"].items() if self._check_date(d, True)
                    }
                }
                for name, st in storage_db.items()
            }
        with open("../../data/terminalDB.json", "r") as f:
            terminal_db = json.load(f)
            terminal_db = {
                name: {
                    "CC": st["CC"],
                    "MonthData": {
                        DataParser._process_date(d): {c: DataParser._process_num(n, c_ns=c_ns) for c, n in ns.items()}
                        for d, ns in st["MonthData"].items() if self._check_date(d)
                    }
                }
                for name, st in terminal_db.items()
            }

        self._logger.info("Reading and preprocessing data is finished.")
        return EGMData(
            cc_list_full,
            consumption_production_assoc,
            export_assoc,
            graph_db,
            prices_assoc,
            storage_db,
            terminal_db
        )


class EGMDataTrainTestSplitter:
    @staticmethod
    def split(data: EGMData, date_split: datetime, mode=1) -> tp.Tuple[EGMData, EGMData]:
        date_train_q = (lambda d: d <= date_split)
        date_test_q = (lambda d: d > date_split)
        delta = relativedelta.relativedelta(months=1)
        date_test_d_q = (lambda d: d > date_split - delta)

        cc_list_full = data.cc_list_full
        cp_assoc_train = {
            cp: {
                cc: {
                    d: data.cp_assoc[cp][cc][d] if date_train_q(d) or cp == "consumption" else LPNan
                    for d in data.cp_assoc[cp][cc]
                }
                for cc in data.cp_assoc[cp]
            }
            for cp in data.cp_assoc
        }
        cp_assoc_test = {
            cp: {
                cc: {d: data.cp_assoc[cp][cc][d] for d in data.cp_assoc[cp][cc] if date_test_q(d)}
                for cc in data.cp_assoc[cp]
            }
            for cp in data.cp_assoc
        }

        export_assoc_train = {
            cc1: {
                cc2: {
                    d: data.export_assoc[cc1][cc2][d] if date_train_q(d) else LPNan for d in data.export_assoc[cc1][cc2]
                }
                for cc2 in data.export_assoc[cc1]
            }
            for cc1 in data.export_assoc
        }
        export_assoc_test = {
            cc1: {
                cc2: {d: data.export_assoc[cc1][cc2][d] for d in data.export_assoc[cc1][cc2] if date_test_q(d)}
                for cc2 in data.export_assoc[cc1]
            }
            for cc1 in data.export_assoc
        }

        graph_db = data.graph_db
        prices_assoc = data.prices_assoc

        k_ = {"workingGasVolume", "injectionCapacity", "withdrawalCapacity"}
        if mode == 0:
            storage_db_train = {
                sto: {
                    "CC": da["CC"],
                    "MonthData": {
                        d: {k: (p if date_train_q or k in k_ else LPNan) for k, p in g.items()}
                        for d, g in da["MonthData"].items()
                    }
                }
                for sto, da in data.storage_db.items()
            }
            terminal_db_train = {
                sto: {
                    "CC": da["CC"],
                    "MonthData": {
                        d: {k: (p if date_train_q or k == "dtrs" else LPNan) for k, p in g.items()}
                        for d, g in da["MonthData"].items()
                    }
                }
                for sto, da in data.terminal_db.items()
            }
        elif mode == 1:
            storage_db_train = data.storage_db
            terminal_db_train = data.terminal_db
        else:
            raise ValueError(f"mode = {mode}")

        storage_db_test = {
            sto: {
                "CC": da["CC"],
                "MonthData": {
                    d: g for d, g in da["MonthData"].items() if date_test_d_q(d)
                }
            }
            for sto, da in data.storage_db.items()
        }
        terminal_db_test = {
            sto: {
                "CC": da["CC"],
                "MonthData": {
                    d: g for d, g in da["MonthData"].items() if date_test_q(d)
                }
            }
            for sto, da in data.terminal_db.items()
        }

        data_train = EGMData(
            cc_list_full,
            cp_assoc_train,
            export_assoc_train,
            graph_db,
            prices_assoc,
            storage_db_train,
            terminal_db_train
        )

        data_test = EGMData(
            cc_list_full,
            cp_assoc_test,
            export_assoc_test,
            graph_db,
            prices_assoc,
            storage_db_test,
            terminal_db_test
        )

        return data_train, data_test
