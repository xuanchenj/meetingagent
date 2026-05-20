from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.models import ParticipantInfo
from livekit.protocol.room import ListRoomsRequest, DeleteRoomRequest, DeleteRoomResponse, ListParticipantsRequest, \
    RoomParticipantIdentity, UpdateParticipantRequest
from livekit.protocol.sip import ListSIPInboundTrunkRequest, ListSIPInboundTrunkResponse, SIPOutboundTrunkInfo, \
    CreateSIPOutboundTrunkRequest, SIPHeaderOptions
from pydantic import BaseModel, Field

load_dotenv()

@asynccontextmanager
async def connect_to_livekit():
    lkc = LiveKitClient()
    try:
        yield lkc
    finally:
        if lkc.c:
            await lkc.close()

class DispatchRuleType(Enum):
    INDIVIDUAL = "dispatch_rule_direct" # 为每个呼叫者分配一个独立房间
    DIRECT = "dispatch_rule_direct" # 直接将呼叫者分配到一个指定房间
    CALLEE = "dispatch_rule_callee" # 根据被叫号码分配房间，房间名为$room_prefix + 被叫号码

class TrunkInfo(BaseModel):
    name: str = Field(..., description="中继名称")
    numbers: list[str] = Field(..., description="号码列表")
    allowed_addresses: list[str] = Field(default=None, description="允许的 IP 地址列表")
    allowed_numbers: list[str] = Field(default=None, description="允许的入站号码列表")
    auth_username: str = Field(default=None, description="用户名")
    auth_password: str = Field(default=None, description="密码")
    metadata: str = Field(default=None, description="元数据")
    krisp_enabled: bool = Field(default=True, description="是否启用 Krisp 噪音消除")

class LiveKitClient:

    async def close(self):
        if self.c:
            await self.c.aclose()

    class Sip:
        def __init__(self, client:api.LiveKitAPI):
            self.c = client

        #region
        #SIP Trunk 相关
        async def create_in_trunk(self, trunk_info:TrunkInfo)-> api.SIPInboundTrunkInfo:
            """
            创建一个 SIP 入站中继
            """
            trunk = api.SIPInboundTrunkInfo(
                name = trunk_info.name,
                numbers = trunk_info.numbers,
                allowed_addresses = trunk_info.allowed_addresses,
                allowed_numbers = trunk_info.allowed_numbers,
                auth_username = trunk_info.auth_username,
                auth_password = trunk_info.auth_password,
                metadata = trunk_info.metadata,
                krisp_enabled = trunk_info.krisp_enabled,
                include_headers= SIPHeaderOptions.SIP_ALL_HEADERS
            )

            request = api.CreateSIPInboundTrunkRequest(
                trunk = trunk
            )
            return await self.c.sip.create_sip_inbound_trunk(request)

        async def delete_in_trunk(self, trunk_id:str) -> api.SIPTrunkInfo:
            """
            删除一个 SIP 入站中继
            Args:
                trunk_id: sip_trunk_id
            """
            request = api.DeleteSIPTrunkRequest(
                sip_trunk_id = trunk_id
            )
            return await self.c.sip.delete_sip_trunk(request)

        async def update_in_trunk(self, trunk_id:str, trunk_info = TrunkInfo) -> api.SIPInboundTrunkInfo:
            """
            更新 SIP 入站中继的字段
            """
            return await self.c.sip.update_sip_inbound_trunk_fields(
                trunk_id = trunk_id,
                name = trunk_info.name,
                numbers = trunk_info.numbers,
                allowed_addresses = trunk_info.allowed_addresses,
                allowed_numbers = trunk_info.allowed_numbers,
                auth_username = trunk_info.auth_username,
                auth_password = trunk_info.auth_password,
                metadata = trunk_info.metadata
            )

        async def replace_in_trunk(self, trunk_id:str, trunk_info:dict) -> api.SIPInboundTrunkInfo:
            """
            替换 SIP 入站中继的字段
            Args:
                trunk_id:
                update_fields:

            """
            trunk = api.SIPInboundTrunkInfo(
                **trunk_info
            )
            return await self.c.sip.update_sip_inbound_trunk(trunk_id, trunk)

        async def list_in_trunks(self) -> list[api.SIPInboundTrunkInfo]:
            """
            列出列表 SIP 入站中继(所有，暂不支持分页和条件查询)
            :param filter:
            :return:
            """
            livekit_api = api.LiveKitAPI()
            resp = await livekit_api.sip.list_sip_inbound_trunk(
                ListSIPInboundTrunkRequest()
            )
            return resp.items

        async def get_in_trunk_by_name(self, trunk_name:str) -> Optional[api.SIPInboundTrunkInfo]:
            """
            根据中继名称获取 SIP 入站中继信息
            """
            trunks = await self.list_in_trunks()
            for trunk in trunks.items:
                if trunk.name == trunk_name:
                    return trunk

        async def get_in_trunk_by_numbers(self, numbers:list[str]) -> Optional[list[api.SIPInboundTrunkInfo]]:
            """
            根据电话号码获取 SIP 入站中继信息
            """
            livekit_api = api.LiveKitAPI()
            trunks = await livekit_api.sip.list_sip_inbound_trunk(
                ListSIPInboundTrunkRequest(numbers=numbers)
            )
            return trunks.items
        #endregion

        #region
        #dispatch 相关
        async def create_dispatch(self,
                                  dispathc_type:DispatchRuleType,
                                  rule_name:str,
                                  trunk_ids:list[str],
                                  agent_name:str,
                                  agent_metadata:str = None,
                                  room_name_info:str = None,
                                  ) -> api.SIPDispatchRuleInfo:
            """
            创建一个 SIP 调度规则
            Args:
                dispathc_type: (DispatchRuleType) 调度规则
                room_name_info: list[str] 房间名称，规则为DispatchRuleType.INDIVIDUAL或者DispatchRuleType.CALLEE 时为房间前缀，DispatchRuleType.DIRECT时为房间全名
                rule_name: str 规则名称
                trunk_ids: list[str] 绑定中继 ID 列表
                agent_name: str 指派的agent名称
                agent_metadata: str 指派的agent元数据
            """
            if dispathc_type == DispatchRuleType.INDIVIDUAL:
                rule = api.SIPDispatchRule(
                    dispatch_rule_individual = api.SIPDispatchRuleIndividual(
                        room_prefix = room_name_info,
                    )
                )
            elif dispathc_type == DispatchRuleType.DIRECT:
                rule = api.SIPDispatchRule(
                    dispatch_rule_direct = api.SIPDispatchRuleDirect(
                        room_name = room_name_info,
                    )
                )
            else:
                rule = api.SIPDispatchRule(
                    dispatch_rule_callee = api.SIPDispatchRuleCallee(
                        room_prefix = room_name_info,
                    )
                )

            request = api.CreateSIPDispatchRuleRequest(
                dispatch_rule = api.SIPDispatchRuleInfo(
                    rule = rule,
                    name = rule_name,
                    trunk_ids = trunk_ids,
                    room_config=api.RoomConfiguration(
                        agents=[api.RoomAgentDispatch(
                            agent_name=agent_name,
                            metadata=agent_metadata,
                        )]
                    )
                )
            )
            return await self.c.sip.create_sip_dispatch_rule(request)

        async def delete_dispatch(self, dispatch_rule_id:str)->api.SIPDispatchRuleInfo:
            """删除一个 SIP 调度规则

            Args:
                dispatch_rule_id: str: SIP 调度规则 ID
            """
            request = api.DeleteSIPDispatchRuleRequest(
                sip_dispatch_rule_id = dispatch_rule_id
            )
            return await self.c.sip.delete_sip_dispatch_rule(request)

        async def update_dispatch(
                self,
                dispatch_rule_id:str,
                trunk_ids: list[str] = None,
                rule_name: str = None,
        ):
            dispatchRule = await self.c.sip.update_sip_dispatch_rule_fields(
                rule_id=dispatch_rule_id,
                name = rule_name,
                trunk_ids = trunk_ids
            )
            return dispatchRule

        async def list_dispatches(self) -> list[api.SIPDispatchRuleInfo]:
            """
            列出所有 SIP 调度规则（暂时不分页）
            """
            resp = await self.c.sip.list_sip_dispatch_rule(
                api.ListSIPDispatchRuleRequest()
            )
            return resp.items

        async def get_dispatch_by_name(self, rule_name:str) -> Optional[api.SIPDispatchRuleInfo]:
            rules = await self.list_dispatches()
            for rule in rules:
                if rule.name == rule_name:
                    return rule

        async def get_dispatch_by_ids(self, dispatch_ids:list[str]) -> Optional[list[api.SIPDispatchRuleInfo]]:
            """
            根据id列表获取 SIP 调度规则信息
            """
            resp = await self.c.sip.list_sip_dispatch_rule(
                api.ListSIPDispatchRuleRequest(
                    dispatch_rule_ids=dispatch_ids
                )
            )
            return resp.items

        async def get_dispatch_by_bind_trunkids(self, trunk_ids:list[str]) -> Optional[list[api.SIPDispatchRuleInfo]]:
            """
            根据绑定的trunkid列表获取 SIP 调度规则信息
            """
            resp = await self.c.sip.list_sip_dispatch_rule(
                api.ListSIPDispatchRuleRequest(
                    trunk_ids=trunk_ids
                )
            )
            return resp.items

        async def create_out_trunk(
                self,
                trunk_name:str,
                numbers:list[str],
                address:str,
                auth_username:str=None,
                password:str=None
        )->SIPOutboundTrunkInfo:
            trunk = SIPOutboundTrunkInfo(
                name = trunk_name,
                address = address,
                numbers = numbers,
                auth_username=auth_username,
                auth_password=password
            )

            request = CreateSIPOutboundTrunkRequest(
                trunk = trunk
            )

            trunk = await self.c.sip.create_sip_outbound_trunk(request)
            return trunk

        async def update_outbound_trunk(self, trunk_id:str, trunk_name=None, address=None, numbers = None, user=None, password=None):
            return await self.c.sip.update_sip_outbound_trunk_fields(
                trunk_id = trunk_id,
                name = trunk_name,
                address = address,
                numbers = numbers,
                auth_username=user,
                auth_password=password
            )


        async def create_sip_participant(
                self,
                *,
                call_number,
                outbound_trunk_id,
                join_room_name,
                participant_identity,
                participant_name,
                headers={},
                sip_number,
                krisp_enabled:bool = False
        ):
            return await self.c.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=outbound_trunk_id,
                    sip_call_to=call_number,
                    room_name=join_room_name,
                    participant_identity=participant_identity,
                    participant_name=participant_name,
                    krisp_enabled=krisp_enabled,
                    headers=headers,
                    sip_number=sip_number,
                    include_headers=SIPHeaderOptions.SIP_ALL_HEADERS,
                    wait_until_answered = True,
                    play_ringtone = True
                )
            )
        #endregion


    class Room:
        def __init__(self, client:api.LiveKitAPI):
            self.c = client

        async def delete_room_byname(self, room_name:str) -> DeleteRoomResponse:
            return await self.c.room.delete_room(DeleteRoomRequest(
                room = room_name,
            ))

        async def get_room_list(self):
            resp = await self.c.room.list_rooms(ListRoomsRequest())
            return resp.rooms



    class Participant:
        def __init__(self, client:api.LiveKitAPI):
            self.c = client

        async def get_room_participants(self, room_name:str) -> list[ParticipantInfo]:
            """
            获取房间内的所有参与者
            """
            res = await self.c.room.list_participants(ListParticipantsRequest(
                room=room_name
            ))
            return res.participants

        async def get_participant_by_identity(self, room_name:str, identity:str) -> ParticipantInfo:
            """
            根据身份获取参与者信息
            """
            return await self.c.room.get_participant(RoomParticipantIdentity(
                room=room_name,
                identity=identity,
            ))


        async def remove_participant_from_room(self, room_name:str, identity:str):
            """
            从房间中移除参与者
            """
            await self.c.room.remove_participant(RoomParticipantIdentity(
                room=room_name,
                identity=identity,
            ))

        async def update_participant_metadata(
                self,
                room_name:str,
                identity:str,
                medatata:str = None
        ):
            """
            更新参与者的元数据
            :param room_name:
            :param identity:
            :param medatata:
            :return:
            """
            await self.c.room.update_participant(UpdateParticipantRequest(
                room=room_name,
                identity=identity,
                metadata=medatata,
            ))

    class Dispatch:
        def __init__(self, client:api.LiveKitAPI):
            self.c = client

        async def dispatch_agent(self):
            self.c.sip



    def __init__(self):
        self.c = api.LiveKitAPI()
        self.sip = LiveKitClient.Sip(self.c)
        self.room = LiveKitClient.Room(self.c)
        self.dispatch = LiveKitClient.Dispatch(self.c)
        self.participant = LiveKitClient.Participant(self.c)
