"""
Backward-compatible collector entrypoint for continuous environments.

At the moment this is a thin alias over EnvDataCollector. Continuous problems
should typically be abstracted first and then collected through the resulting
discrete MDP interface.
"""

from .env_data_collector import EnvDataCollector


class ContinuousEnvDataCollector(EnvDataCollector):
    pass
