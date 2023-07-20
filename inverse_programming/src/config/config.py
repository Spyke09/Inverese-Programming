import inverse_programming.src.lib.copt_pulp as copt_pulp

MESSAGES = False
SOLVER = copt_pulp.COPT_DLL(msg=MESSAGES)

LOG_LEVEL = 0

BIG_M = 10e5
