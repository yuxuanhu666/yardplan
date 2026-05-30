import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Depends
from datetime import datetime
from typing import Dict, List, Optional,Any
from pydantic import BaseModel, Field

from config import service_config
from consul_registration import ConsulRegistration

from pydantic import BaseModel, Field, RootModel
import requests
from fastapi import Request

from algo.dualCycle.dual_cycle_planner import run_dual_cycle_algorithm
from algo.Expertdecking.best_position import Expert_Decking_algorithm
from algo.YardCrane.parallel import run_yard_crane_scheduling
from algo.SpaceAllocation.yardplan import run_plan
from yardplan_core.runtime import run_plan


from algo.QCschedule.qc_scheduling_core import run_qc_scheduling
from yardplanRequest import SpaceCurrent,ShipVisit,BoundItem,UnitDataWrapper

#设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name)

#创建Consul注册对象
consul_registration = ConsulRegistration()

#全局 token 变量
GLOBAL_TOKENS = {}

MODULE_REGISTER_CONFIG = {
    "dualCycle": {
        "cnName": "岸桥边装边卸生成算法",
        "enName": "QC Double-cycling",
        "module": "HL.DualCyle",
        "algoType": 12,
        "version": "V1.0.0",
        "status": "1",
        "url": "http://10.28.120.136:9150/algo/twinCycle/request",
        "note": "华东理工大学-边装边卸算法",
        "isCurrent": 1
    },
    "Expertdecking":{
        "cnName": "专家派位算法",
        "enName": "Expert-decking",
        "module": "HL.Expert_decking",
        "algoType": 6,
        "version": "V1.0.0",
        "status": "1",
        "url": "http://10.28.120.136:9150/module/contr/slot/request",
        "note": "华东理工大学-专家派位算法",
        "isCurrent": 1
    },
    "YcScheduler":{
        "cnName": "场桥智能调度算法",
        "enName": "YC Scheduler",
        "module": "HL.YC_Scheduler",
        "algoType": 9,
        "version": "V1.0.0",
        "status": "1",
        "url": "http://10.28.120.136:9150/module/contr/yc/scheduler/request",
        "note": "华东理工大学-场桥智能调度算法",
        "isCurrent": 1
    },
    "SpacePlan":{
        "cnName": "堆场空间分配算法",
        "enName": "Space-plan",
        "module": "HL.Space_plan",
        "algoType": 9,
        "version": "V1.0.0",
        "status": "1",
        "url": "http://10.28.120.136:9150/module/contr/spaceplan/scheduler/request",
        "note": "华东理工大学-堆场空间分配算法",
        "isCurrent": 1
    },
    "QcScheduler": {
        "cnName": "工班计划算法",
        "enName": "QC Work-shift",
        "module": "HL.QC Work-shift",
        "algoType": 13,
        "version": "V1.0.0",
        "status": "1",
        "url": "http://10.28.120.136:9151/module/contr/qc/scheduler/request",
        "note": "华东理工大学-工班计划算法",
        "isCurrent": 1
    },
    "AgvScheduler": {
        "cnName": "集卡池分配算法",
        "enName": "TT Pooling",
        "module": "HL.TT Pooling",
        "algoType": 14,
        "version": "V1.0.0",
        "status": "1",
        "url": "http://10.28.120.136:9151/module/contr/agv/scheduler/request",
        "note": "华东理工大学-集卡池分配算法",
        "isCurrent": 1
    }
}

def get_token(algo_name: str) -> Optional[str]:
    """
    根据模块名注册算法并获取 token，保存到 GLOBAL_TOKENS 中。
    """
    global GLOBAL_TOKENS

    register_url = "http://10.28.120.217/optimize/register/update"

    if algo_name not in MODULE_REGISTER_CONFIG:
        logger.error(f"未找到模块配置: {algo_name}")
        GLOBAL_TOKENS[algo_name] = None
        return None

    register_data = [MODULE_REGISTER_CONFIG[algo_name]]

    try:
        res = requests.post(register_url, json=register_data, timeout=10)
        res.raise_for_status()

        res_json = res.json()
        token = res_json["data"][0]["token"]

        GLOBAL_TOKENS[algo_name] = token
        logger.info(f"{algo_name} Token 获取成功: {token}")
        return token

    except Exception as e:
        logger.error(f"{algo_name} Token 获取失败: {e}", exc_info=True)
        GLOBAL_TOKENS[algo_name] = None
        return None

def get_module_token(algo_name: str) -> Optional[str]:
    """
    获取模块 token，如果不存在则自动重新获取。
    """
    token = GLOBAL_TOKENS.get(algo_name)
    if not token:
        token = get_token(algo_name)
    return token

def verify_token(algo_name: str, authorization: str = Header(None)):
    """
    校验请求头 Authorization 是否与注册 token 一致
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="缺少 Authorization")

    expected_token = get_module_token(algo_name)

    if not expected_token:
        raise HTTPException(status_code=500, detail="服务未获取到 token")

    if authorization != expected_token:
        raise HTTPException(status_code=403, detail="Token 校验失败")
    
#使用 Lifespan 事件处理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    hello_on_startup()

    for algo_name in MODULE_REGISTER_CONFIG.keys():
        get_token(algo_name)
    consul_registration.register_service()

    yield

    goodbye_on_shutdown()
    consul_registration.deregister_service()


#创建FastAPI应用
app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI service!"}


@app.get("/health")
def health_check():
    """健康检查接口供Consul调用"""
    return "UP"


@app.get("/optimize/test")
def test():
    """测试接口"""
    return {"msg": "success"}


#==========================================
#1. 请求数据模型
#==========================================
class TwinCycleRequest(BaseModel):
    wqKey1: int
    wqKey2: int

#岸桥调度算法请求模型（按文档格式，必填字段正确）
class QcSchedulerRequest(BaseModel):
    vesselVisitKey: int = Field(..., description="船次号")
    shiftQCKeyList: Optional[List[int]] = Field(None, description="人工指派的工班QC Key列表")
    type: int = Field(..., description="类型: 1-adjust功能; 2-create")

#AGV智能调度算法请求模型（按文档格式，必填字段正确）
class AgvSchedulerRequest(BaseModel):
    powKeyList: List[int] = Field(..., description="POW Key列表")

class SpacePlanRequest(BaseModel):
    linekeys: Optional[List[int]] = None
    type: Optional[List[int]] = None

class ExpertDeckingRequest(BaseModel):
    contrIdList: List[str]

class YcSchedulerRequest(BaseModel):
    jobKeyList: List[int] = Field(..., description="作业Key列表")

#==========================================
#2. 响应数据模型
#==========================================
class TwinCycleResponseData(BaseModel):
    errorMsg:str
    dischargeStartPoint: int
    loadStartPoint: int
    subWqId:list[str]

class TwinCycleResponse(BaseModel):
    code: int
    msg: str
    data: TwinCycleResponseData
    success: bool
    btnCode: int

class ExpertDeckingParamDTO(BaseModel):
    wqId: Optional[str] = None
    wiKey: Optional[int] = None
    unitKey: int
    type: Optional[int] = None
    vesselVisitKey: Optional[int] = None
    planPosition: Optional[str] = ""
    targetSlot: str
    targetTP: Optional[str] = ""
    seq: Optional[int] = None
    liftGroupId: Optional[str] = None
    errMsg: Optional[str] = None

class ExpertResponse(BaseModel):
    code: int
    msg: str
    data: List[ExpertDeckingParamDTO]
    success: bool
    btnCode: int

class EqpAssignResultSimple(BaseModel):
    successful: bool = Field(..., description="是否成功")
    wiKey: Optional[int] = Field(None, description="作业指令Key")
    jobKey: int = Field(..., description="作业Key")
    eqpType: int = Field(..., description="设备类型，场桥调度固定填311")
    eqpId: str = Field(..., description="设备ID")
    eqpKey: int = Field(..., description="设备Key")
    bManual: bool = Field(False, description="是否人工指派")
    containerId: str = Field(..., description="箱号")
    failReason: Optional[str] = Field(None, description="失败原因")
    group: Optional[str] = Field(None, description="双箱任务组(jobKey1-jobKey2)")
    qctpFullName: str = Field("", description="岸桥TP全称")
    cancelHTJobKey: List[int] = Field(default_factory=list, description="取消的HT作业Key列表")

class YcSchedulerResponse(BaseModel):
    code: int
    msg: str
    data: List[EqpAssignResultSimple]
    success: bool
    btnCode: int

class SpaceAllocationRangeItem(BaseModel):
    blockId: str = Field(..., description="贝位ID")
    startBayIndex: int = Field(..., description="起始Bay索引")
    endBayIndex: int = Field(..., description="结束Bay索引")
    startStackIndex: int = Field(..., description="起始Stack索引")
    endStackIndex: int = Field(..., description="结束Stack索引")
    startTierIndex: int = Field(..., description="起始Tier索引")
    endTierIndex: int = Field(..., description="结束Tier索引")

class SpaceAllocationFilterItem(BaseModel):
    filterName: str = Field(..., description="过滤器名称")
    isoType: List[str] = Field(default_factory=list, description="ISO箱型代码")
    category: List[int] = Field(default_factory=list, description="集装箱流向")
    pod: List[str] = Field(default_factory=list, description="目的港")
    cattierKind: List[str] = Field(default_factory=list, description="航次")
    tradeCode: List[str] = Field(default_factory=list, description="航线代码")
    freightKind: List[int] = Field(default_factory=list, description="装货方式")
    bReefer: bool = Field(False, description="是否冷藏箱")
    bHazardous: bool = Field(False, description="是否危险品箱")
    bDamage: bool = Field(False, description="是否破损箱")
    bHigh: bool = Field(False, description="是否超高箱")
    bGauge: bool = Field(False, description="是否超标箱")
    ownerCompany: List[str] = Field(default_factory=list, description="箱主")
    lineCompany: List[str] = Field(default_factory=list, description="船公司/运营公司")
    truckCompany: List[str] = Field(default_factory=list, description="集卡公司")
    belongerCompany: List[str] = Field(default_factory=list, description="箱属")
    bDirty: bool = Field(False, description="是否污损")
    weightClass: Optional[int] = Field(None, description="重量等级")
    weightMin: Optional[float] = Field(None, description="最小重量")
    weightMax: Optional[float] = Field(None, description="最大重量")
    workType: Optional[int] = Field(None, description="作业类型")
    bol: List[str] = Field(default_factory=list, description="提单号")
    damageCode: List[str] = Field(default_factory=list, description="危险品类型")

class SpaceAllocationResultItem(BaseModel):
    groupKey: Optional[int] = Field(None, description="分组键值")
    groupId: Optional[int] = Field(None, description="分组ID")
    filter: SpaceAllocationFilterItem = Field(..., description="过滤条件")
    rangeList: List[SpaceAllocationRangeItem] = Field(default_factory=list, description="分配范围列表")

class SpacePlanResponseData(RootModel[List[SpaceAllocationResultItem]]):
    pass

class SpacePlanResponse(BaseModel):
    code: int
    msg: str
    data: SpacePlanResponseData

#岸桥调度响应模型
class ShiftsBreakResult(BaseModel):
    breakKey: int = Field(..., description="休息Key")
    breakName: str = Field(..., description="休息名称")
    breakReason: str = Field(..., description="休息原因")
    startTime: str = Field(..., description="开始时间")
    endTime: str = Field(..., description="结束时间")
    breakType: int = Field(..., description="休息类型")
    duration: int = Field(..., description="持续时间（分钟）")

class AutoShiftsResultSimple(BaseModel):
    shiftKey: int = Field(..., description="工班Key")
    shiftName: str = Field(..., description="工班名称")
    vesselVisitKey: int = Field(..., description="船次Key")
    vesselVisitId: str = Field(..., description="船次ID")
    startTime: str = Field(..., description="开始时间")
    endTime: str = Field(..., description="结束时间")
    duration: int = Field(..., description="持续时间（分钟）")
    powKey: int = Field(..., description="POW Key")
    breakResultList: Optional[List[ShiftsBreakResult]] = Field(None, description="休息时间段列表")

class QcSchedulerResponse(BaseModel):
    code: int
    msg: str
    data: List[AutoShiftsResultSimple]
    success: bool
    btnCode: int

#AGV响应模型
class TTPoolingResultDTOForHL(BaseModel):
    powKeyList: Dict[int, List[str]] = Field(..., description="<powKey, List<htId>> 共享pool时，给出每个pow对应的htid")

class AgvSchedulerResponse(BaseModel):
    code: int
    msg: str
    data: List[TTPoolingResultDTOForHL]
    success: bool
    btnCode: int

#==========================================
#3. 接口实现
#==========================================
@app.post("/algo/twinCycle/request", response_model=TwinCycleResponse)
def algo_twin_cycle_request(
    request: TwinCycleRequest,
    _: Any = Depends(lambda authorization=Header(None): verify_token("dualCycle", authorization))
):
    logger.info("========== 收到边装边卸算法请求 ==========")
    logger.info(f"请求参数: wqKey1={request.wqKey1}, wqKey2={request.wqKey2}")
    try:
        result = run_dual_cycle_algorithm(
            get_module_token("dualCycle"),
            request.wqKey1,
            request.wqKey2
        )

        return {
            "code": 200,
            "msg": "操作成功",
            "success": True,
            "data": {
                "errorMsg": result["errorMsg"],
                "loadStartPoint": result["loadStartPoint"],
                "dischargeStartPoint": result["dischargeStartPoint"],
                "subWqId": result["subWqId"]
            },
            "btnCode": 0
        }

    except Exception as e:
        logger.error(f"算法执行异常: {e}", exc_info=True)
        return {
            "code": 500,
            "msg": f"算法执行失败: {str(e)}",
            "success": False,
            "data": {
                "errorMsg": str(e),
                "loadStartPoint": 0,
                "dischargeStartPoint": 0,
                "subWqId": []
            },
            "btnCode": 0
        }

@app.post("/module/contr/slot/request", response_model=ExpertResponse)
async def algo_slot_request(
    request: ExpertDeckingRequest,
    _: Any = Depends(lambda authorization=Header(None): verify_token("Expertdecking", authorization))
):
    logger.info("========== 收到专家派位算法请求 ==========")
    logger.info(f"请求参数: contrIdList={request.contrIdList}")

    try:
        token = get_module_token("Expertdecking")
        data = Expert_Decking_algorithm(token, request.model_dump())

        return {
            "code": 200,
            "msg": "操作成功",
            "data": data,
            "success": True,
            "btnCode": 0
        }

    except Exception as e:
        logger.error(f"专家派位算法执行异常: {e}", exc_info=True)
        return {
            "code": 500,
            "msg": f"算法执行失败: {str(e)}",
            "data": [],
            "success": False,
            "btnCode": 0
        }

@app.post("/module/contr/yc/scheduler/request", response_model=YcSchedulerResponse)
async def algo_YcScheduler_request(
    request: YcSchedulerRequest,
    _: Any = Depends(lambda authorization=Header(None): verify_token("YcScheduler", authorization))
):
    logger.info("========== 收到场桥调度算法请求 ==========")
    logger.info(f"请求参数: jobKeyList={request.jobKeyList}")

    try:
        token = get_module_token("YcScheduler")
        result_list = run_yard_crane_scheduling(token, request.jobKeyList)

        return {
            "code": 200,
            "msg": "操作成功",
            "data": result_list,
            "success": True,
            "btnCode": 0
        }

    except Exception as e:
        logger.error(f"场桥调度算法执行异常: {e}", exc_info=True)
        return {
            "code": 500,
            "msg": f"算法执行失败: {str(e)}",
            "data": [],
            "success": False,
            "btnCode": 0
        }

#岸桥接口（已修正，接收正确参数）
@app.post("/module/contr/qc/scheduler/request", response_model=QcSchedulerResponse)
async def algo_qc_scheduler_request(
    request: QcSchedulerRequest,
    _: Any = Depends(lambda authorization=Header(None): verify_token("QcScheduler", authorization))
):
    logger.info("========== 收到岸桥工班调度算法请求 ==========")
    logger.info(f"请求参数: vesselVisitKey={request.vesselVisitKey}, shiftQCKeyList={request.shiftQCKeyList}, type={request.type}")

    return {
        "code": 200,
        "msg": "操作成功",
        "success": True,
        "btnCode": 0,
        "data": [
            {
                "shiftKey": 5001,
                "shiftName": "早班",
                "vesselVisitKey": 3001,
                "vesselVisitId": "VV2026041601",
                "startTime": "2026-04-16 08:00:00",
                "endTime": "2026-04-16 16:00:00",
                "duration": 480,
                "powKey": 6001,
                "breakResultList": [
                    {
                        "breakKey": 7001,
                        "breakName": "午餐休息",
                        "breakReason": "正常休息",
                        "startTime": "2026-04-16 12:00:00",
                        "endTime": "2026-04-16 12:30:00",
                        "breakType": 1,
                        "duration": 30
                    },
                    {
                        "breakKey": 7002,
                        "breakName": "茶歇",
                        "breakReason": "正常休息",
                        "startTime": "2026-04-16 15:00:00",
                        "endTime": "2026-04-16 15:15:00",
                        "breakType": 2,
                        "duration": 15
                    }
                ]
            }
        ]
    }

#AGV接口（已修正，接收正确参数）
@app.post("/module/contr/agv/scheduler/request", response_model=AgvSchedulerResponse)
async def algo_agv_scheduler_request(
    request: AgvSchedulerRequest,
    _: Any = Depends(lambda authorization=Header(None): verify_token("AgvScheduler", authorization))
):
    logger.info("========== 收到集卡池分配算法请求 ==========")
    logger.info(f"请求参数: powKeyList={request.powKeyList}")

    return {
        "code": 200,
        "msg": "操作成功",
        "success": True,
        "btnCode": 0,
        "data": [
            {
                "powKeyList": {
                    5001: ["HT001", "HT002", "HT003"],
                    5002: ["HT004", "HT005"],
                    5003: ["HT006"]
                }
            },
            {
                "powKeyList": {
                    5004: ["HT007", "HT008"],
                    5005: ["HT009", "HT010", "HT011"]
                }
            }
        ]
    }

@app.post("/module/contr/spaceplan/scheduler/request", response_model=SpacePlanResponse)
def algo_space_plan_request(
    request: SpacePlanRequest,
    _: Any = Depends(lambda authorization=Header(None): verify_token("SpacePlan", authorization))):

    logger.info("========== 收到堆存计划算法请求 ==========")
    logger.info(f"请求参数: linekeys={request.linekeys},type={request.type}")

    try:
        token = get_module_token("SpacePlan")
        result_list = run_plan(token,request.linekeys,request.type)
        return {
            "code": 200,
            "msg": "success",
            "data": result_list,   
            "success": True,
        }
    except Exception as e:
        logger.error(f"算法执行异常: {e}", exc_info=True)
        return {
            "code": 500,
            "msg": f"算法执行失败: {str(e)}",
            "data": [],
            "success": False,
            "btnCode": 0
        }

def hello_on_startup():
    print("")
    print("/-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--/")
    print("/----     FastAPI APPLICATION STARTING...     ----/")
    print(f"/----          {service_config.SERVICE_NAME}            ----/")
    print(f"/----       {datetime.now()}        ----/")
    print("/--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-/")
    print("")

def goodbye_on_shutdown():
    print("")
    print("/-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--/")
    print("/----  FastAPI APPLICATION SHUTTING DOWN...   ----/")
    print(f"/----       {datetime.now()}        ----/")
    print("/--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-/")
    print("")

if name == "main":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=service_config.SERVICE_PORT,
        reload=False
    )