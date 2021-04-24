from aip import AipSpeech

""" 你的 APPID AK SK """
APP_ID = '23990390'
API_KEY = 'c4GfQcmRtAPtqStD6NHIS1tf'
SECRET_KEY = 'SDMzr5peouYIc2lZEnSe0Yi4R1HeLwk2'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 读取文件
def get_file_content(filep):
    with open(filep, 'rb') as fp:
        return fp.read()
# 识别本地文件
def get_text():
    result = client.asr(get_file_content(r"D:\王晨E\pythonProject\for_test\mysite\static\files\疑问句.wav"), 'wav', 16000, {
    'dev_pid': 1537})
    print(result)
    #text = result['result'][0]
    #return text
    #print(text)

#print(get_text())
get_text()

# import requests
# import json
# import os
# import base64
#
# # 设置应用信息
# baidu_server = "https://openapi.baidu.com/oauth/2.0/token?"
# grant_type = "client_credentials"
# client_id = "c4GfQcmRtAPtqStD6NHIS1tf"  # 填写API Key
# client_secret = "SDMzr5peouYIc2lZEnSe0Yi4R1HeLwk2"  # 填写Secret Key
#
# # 合成请求token的URL
# url = baidu_server + "grant_type=" + grant_type + "&client_id=" + client_id + "&client_secret=" + client_secret
# print("url:", url)
#
# # 获取token
# res = requests.post(url)
# print(res.text)
# token = json.loads(res.text)["access_token"]
# print(token)
# # 24.b891f76f5d48c0b9587f72e43b726817.2592000.1524124117.282335-10958516
#
# # 设置格式
# RATE = "16000"
# FORMAT = "wav"
# CUID = "wate_play"
# DEV_PID = "1536"
#
# # 以字节格式读取文件之后进行编码
# with open(r"D:\王晨E\pythonProject\for_test\mysite\static\files\疑问句.wav", "rb") as f:
#     speech = base64.b64encode(f.read()).decode('utf8')
# size = os.path.getsize(r"D:\王晨E\pythonProject\for_test\mysite\static\files\疑问句.wav")
# headers = {'Content-Type': 'application/json'}
# url = "https://vop.baidu.com/server_api"
# data = {
#
#     "format": FORMAT,
#     "rate": RATE,
#     "dev_pid": DEV_PID,
#     "speech": speech,
#     "cuid": CUID,
#     "len": size,
#     "channel": 1,
#     "token": token,
# }
#
# req = requests.post(url, json.dumps(data), headers)
# result = json.loads(req.text)
# print(result["result"][0])
