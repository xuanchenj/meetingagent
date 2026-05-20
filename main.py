from livekit.agents import (
    cli
)

from agent.meeting_agent import server

if __name__ == '__main__':
    cli.run_app(server)