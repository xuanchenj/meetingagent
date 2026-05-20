import asyncio

from dotenv import load_dotenv

from livekit_client import connect_to_livekit, DispatchRuleType

load_dotenv(".env.local")


async def create_dispath():
    async with connect_to_livekit() as lk:
        print(await lk.sip.create_dispatch(
            dispathc_type = DispatchRuleType.INDIVIDUAL,
            rule_name = "test",
            trunk_ids = ["ST_qSSGtAKsciFs"],
            agent_name = "voice_demo",
            room_name_info = "call"
        ))

async def create_outbound_trunk():
    async with connect_to_livekit() as lk:
        await lk.sip.create_out_trunk(
            trunk_name = "takeout",
            numbers = ["*"],
            address = "47.102.85.173:32766"
            # auth_username = "6721",
            # password="usPKpKY7Fq"
        )

async def create_outbound_trunk2():
    async with connect_to_livekit() as lk:
        await lk.sip.create_out_trunk(
            trunk_name = "ali_tran_to_human",
            numbers = ["*"],
            address = "139.196.137.211:5160"
        )
async def create_trunk():
    async with connect_to_livekit() as lk:
        trunk = TrunkInfo(
            name = "ten_test_num",
            numbers = ["700002"]
        )
        print(await lk.sip.create_in_trunk(trunk))

if __name__ == '__main__':
    asyncio.run(create_outbound_trunk())