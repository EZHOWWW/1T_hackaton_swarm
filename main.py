import asyncio
from config import LOG_LEVEL
from services.logger import set_logger_config


async def start_websocket():
    from solution.simulation import Simulation
    from solution.swarm import Swarm

    while True:
        sim = Simulation()
        await sim.connect_to_server()
    
        # here is solution
        swarm = Swarm(sim)
        swarm.start()
    
        sim.close_connection()

        if input("To exit print 'q'") == "q":
            break


if __name__ == "__main__":
    set_logger_config(LOG_LEVEL)
    asyncio.run(start_websocket())
