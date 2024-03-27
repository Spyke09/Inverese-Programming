import logging
from datetime import datetime

import numpy as np

from unique_bilevel_programming_cplex.src.egm import data_parser
from unique_bilevel_programming_cplex.src.egm import data_spitter
from unique_bilevel_programming_cplex.src.egm import egm

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logger = logging.getLogger("Test")


    def egm_test_1():
        a = 2
        b = 0
        dates = [datetime(2019, i, 1) for i in range(1, a + 1)]
        dates_test = dates[b + 1:]
        parser = data_parser.DataParser(dates)

        data = parser.get_data()

        for mode in range(1, 2):
            logger.info(f"Mode {mode}.")

            train_data, test_data = data_spitter.EGMDataTrainTestSplitter.split(data, dates[b], mode=mode)
            model = egm.EGRMinCostFlowModel(
                big_m=1e8,
                eps=1e-2,
                price_lag=1,
                first_unique=True,
                gap=0.001,
                # time_for_optimum=100
            )

            model.fit(train_data, dates)
            model.write_results(f"out/res_2019_mode_{mode}.json")

            smape = (lambda x, y: 200 / x.shape[0] * np.sum(np.abs(x - y) / (0.1 + np.abs(x) + np.abs(y))))

            x_true = model.get_x_0(test_data, dates_test)
            x_pred = model.predict_x(x_true.keys())

            x_true_np = np.array([x_true[i] for i in x_true.keys()])
            x_pred_np = np.array([x_pred[i] for i in x_true.keys()])
            logger.info(f"x smape: {smape(x_true_np, x_pred_np)}")

            ub_train = model.known_ub
            b_true = {i: i.b_coef for i in ub_train}
            b_pred = model.predict_b(b_true.keys())

            b_true_np = np.array([b_true[i] for i in b_true.keys()])
            b_pred_np = np.array([b_pred[i] for i in b_true.keys()])
            logger.info(f"u smape: {smape(b_true_np, b_pred_np)}")


    egm_test_1()
