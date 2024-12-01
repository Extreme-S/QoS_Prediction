import pandas as pd
import matplotlib.pyplot as plt


def plot_response_time():
    # 读取数据, 筛选指定的 UserID 和 ServiceID
    rtdata_path = '/Users/lyw/projects/ECNU/WSRec/data/dataset2/rtdata.txt'
    tpdata_path = '/Users/lyw/projects/ECNU/WSRec/data/dataset2/tpdata.txt'
    rtdata = pd.read_csv(rtdata_path, sep=' ', header=None, names=['UserID', 'ServiceID', 'TimeSliceID', 'ResponseTime'])
    tpdata = pd.read_csv(tpdata_path, sep=' ', header=None, names=['UserID', 'ServiceID', 'TimeSliceID', 'Throughput'])

    for user_id in range(58, 59):
        for service_id in range(50, 51):
            filtered_rtdata = rtdata[(rtdata['UserID'] == user_id) & (rtdata['ServiceID'] == service_id)]
            filtered_tpdata = tpdata[(tpdata['UserID'] == user_id) & (tpdata['ServiceID'] == service_id)]
            if filtered_rtdata.empty or filtered_tpdata.empty:
                print(f"No data found for UserID: {user_id} and ServiceID: {service_id}")
                continue

            # 绘制图像
            plt.figure(figsize=(10, 6))
            plt.plot(filtered_rtdata['TimeSliceID'], filtered_rtdata['ResponseTime'], marker='o', label='Response Time')
            # plt.plot(filtered_tpdata['TimeSliceID'], filtered_tpdata['Throughput'], marker='x', label='Throughput')
            plt.title(f'QoS for User {user_id} and Service {service_id}')
            plt.xlabel('Time Slice ID')
            plt.ylabel('QoS Metrics')
            # plt.grid(True)
            plt.legend()  # 显示图例
            plt.xticks(filtered_rtdata['TimeSliceID'])  # 设置横坐标刻度
            plt.tight_layout()
            plt.ylim(0)
            plt.savefig(f'./response_time_user_{user_id}_service_{service_id}.png')
            # plt.show()


if __name__ == "__main__":
    plot_response_time()
