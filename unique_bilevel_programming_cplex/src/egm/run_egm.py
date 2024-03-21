import logging
from datetime import datetime

from unique_bilevel_programming_cplex.src.egm import data_parser
from unique_bilevel_programming_cplex.src.egm import egm

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    model = egm.EGRMinCostFlowModel(
        data_parser.DataParser.get_data(),
        [datetime(2019, i, 1) for i in range(1, 13)],
        big_m=1e8,
        eps=1e-2,
        lag=12,
        force_b=False
    )

    model.setup()

    solution = model.solve()
    print(solution)
