import requests
import json
import os
from typing import Optional, Dict
from config import service_config


def _url(path: str) -> str:
    return f"{service_config.TOS_BASE_URL}/{path.lstrip('/')}"

def blocklist_receive(token: object,save_path: str = "./data217/block_data.json") -> object:
    """
    初始化 堆场箱区结构，每个箱区的坐标、贝、列、层数
    """
    # # ======================== 接收数据 ============================
    # url_block = _url("/yard/block/query")
    # headers = {
    #     "Authorization": token,
    #     "Content-Type": "application/json"
    # }
    # response = requests.get(url_block, headers=headers)
    # if not response.ok:
    #    raise Exception(f"获取箱区数据失败，状态码：{response.status_code}")

    # data = response.json()

    # # 保存完整返回结果到本地 json 文件
    # with open(save_path, "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=2)

    # return response.json().get("data")
    # # with open('./data/blocklist.json', 'r',encoding="utf-8") as file:
    # #     data = json.load(file).get('data')
    # # return data



def ShipVisitList_receive(token):
    """
    船舶访问计划获取。
    """
    #======================== 接收数据 ============================
    url_block = _url("/ship/visit/query")
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }
    response = requests.get(url_block, headers=headers)
    if not response.ok:
        raise Exception(f"获取船舶访问计划数据失败，状态码：{response.status_code}")

    return response.json()


    # with open('./data/SHIPVISITID_TEST1220.json', 'r',encoding="utf-8") as file:
    #     data = json.load(file).get('data')
    # return data

def SpaceCurrent_receive(token):
    """
    在场箱信息获取。
    """
    # ======================== 接收数据 ============================
    url_block = _url("/inve/unit/query")
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }
    response = requests.get(url_block, headers=headers)
    if not response.ok:
        raise Exception(f"获取未占用箱位数据失败，状态码：{response.status_code}")

    return response.json()
    

def BoundList_receive(token):
    """
        初始化 ZoneConfig，通过接口获取箱区信息。
        """
    # ======================== 接收数据 ============================
    url_block = _url("/inve/unit/query")
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }
    response = requests.get(url_block, headers=headers)
    if not response.ok:
        raise Exception(f"获取所有箱子数据失败，状态码：{response.status_code}")

    return response.json()


def SpaceLocation_receive(token):
    """
    初始化 ZoneConfig，通过接口获取箱区信息。
    """
    # ======================== 接收数据 ============================
    url_block = _url("/inve/unit/query")
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }
    response = requests.get(url_block, headers=headers)
    if not response.ok:
        raise Exception(f"获取所有箱子数据失败，状态码：{response.status_code}")

    return response.json()


# def new_container_receive(token):
#     """
#     初始化 ZoneConfig，通过接口获取箱区信息。
#     """
#     # ======================== 接收数据 ============================
#     url_block = "http://10.28.120.165/inve/unit/query"
#     headers = {
#         "Authorization": token,
#         "Content-Type": "application/json"
#     }
#     response = requests.get(url_block, headers=headers)
#     if not response.ok:
#         raise Exception(f"获取新集装箱数据失败，状态码：{response.status_code}")

#     return response.json().get("data")

# def weight_config_receive(token):
#     """
#     初始化 ZoneConfig，通过接口获取箱区信息。
#     """
#     # ======================== 接收数据 ============================
#     url_block = "http://10.28.120.165/yard/block/query"
#     headers = {
#         "Authorization": token,
#         "Content-Type": "application/json"
#     }
#     response = requests.get(url_block, headers=headers)
#     if not response.ok:
#         raise Exception(f"获取权重数据失败，状态码：{response.status_code}")

#     return response.json().get("data")

# def gate_position_params_receive(token):
#     """
#     初始化 ZoneConfig，通过接口获取箱区信息。
#     """
#     # ======================== 接收数据 ============================
#     # url_block = ""
#     # headers = {
#     #     "Authorization": token,
#     #     "Content-Type": "application/json"
#     # }
#     # response = requests.get(url_block, headers=headers)
#     # if not response.ok:
#     #     raise Exception(f"获取坐标数据失败，状态码：{response.status_code}")
#     #
#     # return response.json().get("data")

#     with open('Position.json', 'r', encoding='utf-8') as file:
#         data = json.load(file).get('data')
#     return data

# def qc_position_params_receive(token):
#     """
#     初始化 ZoneConfig，通过接口获取箱区信息1。
#     """
#     # ======================== 接收数据 ============================
#     url_block = "http://10.28.120.165/yard/block/query"
#     headers = {
#         "Authorization": token,
#         "Content-Type": "application/json"
#     }
#     response = requests.get(url_block, headers=headers)
#     if not response.ok:
#         raise Exception(f"获取坐标数据失败，状态码：{response.status_code}")

#     return response.json().get("data")

# def cost_params_receive(token):
#     """
#     初始化 ZoneConfig，通过接口获取箱区信息。
#     """
#     # ======================== 接收数据 ============================
#     url_block = "http://10.28.120.165/yard/block/query"
#     headers = {
#         "Authorization": token,
#         "Content-Type": "application/json"
#     }
#     response = requests.get(url_block, headers=headers)
#     if not response.ok:
#         raise Exception(f"获取成本系数数据失败，状态码：{response.status_code}")

#     return response.json().get("data")

# def computation_params_receive(token):
#     """
#     初始化 ZoneConfig，通过接口获取箱区信息。
#     """
#     # ======================== 接收数据 ============================
#     url_block = "http://10.28.120.165/yard/block/query"
#     headers = {
#         "Authorization": token,
#         "Content-Type": "application/json"
#     }
#     response = requests.get(url_block, headers=headers)
#     if not response.ok:
#         raise Exception(f"获取惩罚系数数据失败，状态码：{response.status_code}")

#     return response.json().get("data")

# def ship_block_list_receive(token: str, visit_id: str) -> Optional[Dict]:
#     """
#     根据船次ID获取船舶物理信息，并保存每步结果到 ./data 文件夹
#     Args:
#         token: 授权token
#         visit_id: 船次ID (如 "MAGNA25003")
#     Returns:
#         船舶物理结构信息，失败返回None
#     """
#     headers = {
#         "Authorization": token,
#         "Content-Type": "application/json"
#     }

#     # 确保 data 文件夹存在
#     os.makedirs('./data', exist_ok=True)

#     try:
#         # Step 1: 查询船次信息，获取 shipKey
#         visit_url = "http://10.28.120.165/ship/visit/query"
#         visit_resp = requests.post(visit_url, headers=headers, json={"visitId": visit_id})

#         if not visit_resp.ok:
#             print(f"查询船次失败，状态码：{visit_resp.status_code}")
#             return None

#         visit_json = visit_resp.json()

#         # 保存船次信息
#         with open(f'./data/ship_visit_{visit_id}.json', 'w', encoding='utf-8') as f:
#             json.dump(visit_json, f, ensure_ascii=False, indent=2)
#         print(f"已保存: ./data/ship_visit_{visit_id}.json")

#         # 从 shipVisitList 中查找匹配的 visit_id
#         ship_visit_list = visit_json.get('data', {}).get('shipVisitList', [])

#         if not ship_visit_list:
#             print(f"船次列表为空")
#             return None

#         # 查找 id 匹配的记录
#         ship_key = None
#         for visit in ship_visit_list:
#             if visit.get('id') == visit_id:
#                 ship_key = visit.get('shipKey')
#                 print(f"找到船次 {visit_id}, shipKey: {ship_key}, 船名: {visit.get('shipName')}")
#                 break

#         if not ship_key:
#             print(f"未找到船次 {visit_id} 对应的 shipKey")
#             return None

#         # Step 2: 查询船舶物理结构
#         block_url = "http://10.28.120.165/ship/block/query"
#         block_resp = requests.post(block_url, headers=headers, json={"shipKey": ship_key})

#         if not block_resp.ok:
#             print(f"查询船舶结构失败，状态码：{block_resp.status_code}")
#             return None

#         block_json = block_resp.json()

#         # 保存船舶物理结构
#         with open(f'./data/ship_block_{visit_id}.json', 'w', encoding='utf-8') as f:
#             json.dump(block_json, f, ensure_ascii=False, indent=2)
#         print(f"已保存: ./data/ship_block_{visit_id}.json")

#         return block_json.get('data')

#     except Exception as e:
#         print(f"获取船舶信息异常：{e}")
#         return None


# if __name__ == '__main__':
#     data2 = [{
#         "cnName": "岸桥边装边卸生成算法",
#         "enName": "Dual-cycle",
#         "module": "HL.Dual_cycle",
#         "algoType": 12,
#         "version": "V1.0",
#         "status": "1",
#         "url": "127.0.0.1",
#         "note": "说明描述。“华理岸桥边装边卸生成算法”",
#         "isCurrent": 1
#     }]
#     # # token接口地址
#     url = service_config.ALGO_REGISTER_URL

#     # 返回值
#     res = requests.post(url, json=data2)
#     # 获取token
#     token = res.json()['data'][0]['token']

#     print(token)
#     print(query_wi_list(token))

