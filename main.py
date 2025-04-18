import asyncio
import json

from config import LOG_LEVEL
from services.logger import set_logger_config


async def start_websocket():
    from algorithm.fly import connection
    from algorithm.PID import concat_engine, concat_engines
    from solution.simulation import Simulation
    from solution.swarm import Swarm

    sim = Simulation()
    await sim.connect_to_server()
    sim.get_drone_info(0)

    swarm = Swarm(sim)

    swarm.start()

    sim.close_connection()


if __name__ == "__main__":
    set_logger_config(LOG_LEVEL)
    asyncio.run(start_websocket())
