import typing as tp
from datetime import datetime

from dateutil import relativedelta

from unique_bilevel_programming_cplex.src.base.common import LPNan
from unique_bilevel_programming_cplex.src.egm.data_parser import EGMData


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
