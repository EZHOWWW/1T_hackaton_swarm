import asyncio
from config import LOG_LEVEL
from services.logger import set_logger_config


async def start_websocket():
    from solution.simulation import Simulation
    from solution.swarm import Swarm

    sim = Simulation()
    await sim.connect_to_server()
    
    while True:
        # here is solution
        swarm = Swarm(sim)
        swarm.start()
        if input("To exit print 'q'") == "q":
            break
        else:
            sim.connection.send_data("restartScene")
            print(sim.connection.receive_data())

    sim.close_connection()


if __name__ == "__main__":
    set_logger_config(LOG_LEVEL)
    asyncio.run(start_websocket())
