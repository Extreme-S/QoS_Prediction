import math


def haversine(lat1, lon1, lat2, lon2):
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # 计算纬度和经度的差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # 哈弗辛公式
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # 地球平均半径，单位为米
    r = 6371000

    # 计算距离
    distance = c * r
    return distance
