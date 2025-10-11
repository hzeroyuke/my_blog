## 1. Elastic Search的安装

我在Ubuntu系统中安装Elastic Search，我选择使用基于Docker的形式来安装Elastic search

![](Pasted%20image%2020251009104458.png)

安装的结果如上

接下来是启动的截图，我们使用如下命令启动

```
docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -it --name es01 docker.elastic.co/elasticsearch/elasticsearch:8.10.4
```


![](Pasted%20image%2020251009105552.png)


## 2. 推荐原因索引

首先我们使用Python来编写ES相关脚本，第一步是导入相关的包 ``pip install "elasticsearch>=8.0.0" pandas openpyxl xlrd`` 

然后因为Elastic Search的docker强制使用https连接，因此需要在docker内部配置一下CA证书，使用如下命令

```bash
docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt ./http_ca.crt
```


因为ES原生支持的中文分词比较糟糕，要做正确的中文索引，需要加载一个IK分词器



代码如下

结果屏幕如下

![](Pasted%20image%2020251010212446.png)

## 3. 找出高月售的店家

代码如下

```python
from elasticsearch import Elasticsearch
import sys
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# --- 连接到 Elasticsearch ---
try:
    es = Elasticsearch(
        ["https://localhost:9200"],
        basic_auth=("elastic", "SMsFXe6WOqTh0-jvXY*v"),
        verify_certs=False,
        request_timeout=30
    )
    print("成功连接到 Elasticsearch\n")
    
except Exception as e:
    print(f"连接失败: {e}", file=sys.stderr)
    sys.exit(1)

# --- 查询月售>=500的商家 ---
INDEX_NAME = "restaurants"

print("="*60)
print("查询月售额 >= 500 的所有商家")
print("="*60)

query = {
    "query": {
        "range": {
            "月售": {
                "gte": 500  # greater than or equal to 500
            }
        }
    },
    "_source": ["商家名称", "月售", "月售原始"],
    "sort": [
        {"月售": {"order": "desc"}}  # 按月售降序排列
    ],
    "size": 1000  # 获取所有匹配结果
}

try:
    response = es.search(index=INDEX_NAME, body=query)
    
    total = response['hits']['total']['value']
    print(f"\n共找到 {total} 家月售额 >= 500 的商家\n")
    
    if total > 0:
        for i, hit in enumerate(response['hits']['hits'], 1):
            print(f"{i}. {hit['_source']['商家名称']} - 月售: {hit['_source']['月售']} (原始值: {hit['_source']['月售原始']})")
    else:
        print("未找到月售额 >= 500 的商家。")
    
    print("="*60)
    
except Exception as e:
    print(f"查询错误: {e}", file=sys.stderr)
    sys.exit(1)
```

结果截屏如下


![](Pasted%20image%2020251010213043.png)

![](Pasted%20image%2020251010213056.png)
![](Pasted%20image%2020251010213125.png)


## 4. 查询推荐

使用Openrouter中的Gemini 2.5 flash来进行处理，获得了一系列的推荐菜品，由于内容较多，节选截屏如下，详情可以看附件

![](Pasted%20image%2020251010214211.png)


相应的代码如下

``` python

from elasticsearch import Elasticsearch
from openai import OpenAI
import sys
import json

# --- 连接到 Elasticsearch ---
try:
    es = Elasticsearch(
        ["https://localhost:9200"],
        basic_auth=("elastic", "SMsFXe6WOqTh0-jvXY*v"),
        verify_certs=False,
        request_timeout=30
    )
    print("成功连接到 Elasticsearch\n")
    
except Exception as e:
    print(f"连接失败: {e}", file=sys.stderr)
    sys.exit(1)

# --- 连接到 OpenRouter ---
# 请替换为你的 OpenRouter API Key
OPENROUTER_API_KEY = "your-openrouter-api-key-here"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="API_KEY",
)

# --- 从 Elasticsearch 获取所有商家数据 ---
INDEX_NAME = "restaurants"

print("正在从 Elasticsearch 获取所有商家数据...\n")

query = {
    "query": {
        "match_all": {}
    },
    "_source": ["商家名称", "被推荐原因"],
    "size": 1000
}

response = es.search(index=INDEX_NAME, body=query)
all_restaurants = response['hits']['hits']

print(f"共获取 {len(all_restaurants)} 条商家数据\n")

# --- 使用大模型分析推荐原因 ---
print("="*60)
print("使用大模型分析推荐原因中提到菜品的商家")
print("="*60)
print()

# 构建推荐原因列表
reasons_data = []
for hit in all_restaurants:
    reasons_data.append({
        "商家名称": hit['_source']['商家名称'],
        "被推荐原因": hit['_source']['被推荐原因']
    })

# 分批处理（每次处理50条，避免token限制）
batch_size = 50
results = []

for i in range(0, len(reasons_data), batch_size):
    batch = reasons_data[i:i+batch_size]
    
    print(f"正在分析第 {i+1}-{min(i+batch_size, len(reasons_data))} 条记录...")
    
    prompt = f"""请分析以下商家的推荐原因，找出那些因为喜欢某个或某些菜品而推荐该商家的记录。

判断标准：
1. 推荐原因中明确提到了具体的菜品名称（如：青椒皮蛋、猪脚饭、鸡杂、美蛙、抄手、羊肉汤等）
2. 推荐原因中表达了对某道菜的喜爱或好评
3. 不包括只是笼统说"好吃"、"美味"、"不错"等没有具体菜品的推荐

数据：
{json.dumps(batch, ensure_ascii=False, indent=2)}

请以JSON格式返回符合条件的记录，格式如下：
{{
  "matches": [
    {{
      "商家名称": "xxx",
      "被推荐原因": "xxx",
      "提到的菜品": ["菜品1", "菜品2"]
    }}
  ]
}}

只返回JSON，不要其他说明文字。"""

    try:
        completion = client.chat.completions.create(
            model="google/gemini-2.5-flash",  # 使用Claude模型
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
        )
        
        response_text = completion.choices[0].message.content
        
        # 提取JSON（处理可能的markdown代码块）
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        if "matches" in result and len(result["matches"]) > 0:
            results.extend(result["matches"])
        
    except Exception as e:
        print(f"  分析出错: {e}")
        continue

print()
print("="*60)
print(f"分析完成！共找到 {len(results)} 条因喜欢菜品而推荐的记录")
print("="*60)
print()

# --- 输出结果 ---
if len(results) > 0:
    for i, item in enumerate(results, 1):
        print(f"{i}. 商家名称: {item['商家名称']}")
        print(f"   被推荐原因: {item['被推荐原因']}")
        print(f"   提到的菜品: {', '.join(item['提到的菜品'])}")
        print()
else:
    print("未找到因喜欢菜品而推荐的商家。")

print("="*60)

# --- 保存结果到文件 ---
output_file = "菜品推荐商家.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存到 {output_file}")
```