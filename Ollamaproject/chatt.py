from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='qwen3:4b', messages=[
  {
    'role': 'user',
    'content': '你是谁？',
  },
])
# 打印响应内容
print(response['message']['content'])
# 或者直接访问响应对象的字段
#print(response.message.content)