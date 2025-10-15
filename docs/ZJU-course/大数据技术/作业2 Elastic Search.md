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

我们从github上下载好zip文件之后，导入到docker中

``` bash
docker cp elasticsearch-analysis-ik-8.10.4.zip d38f84229931:/usr/share/elasticsearch/
```

代码如下

```python
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
import sys
import os
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

INDEX_NAME = "restaurants"

INDEX_MAPPING = {
    "properties": {
        "商家名称": {
            "type": "text",
            "analyzer": "ik_max_word",
            "search_analyzer": "ik_smart"
        },
        "地址": {
            "type": "text",
            "analyzer": "ik_max_word",
            "search_analyzer": "ik_smart"
        },
        "所在省份": {"type": "keyword"},
        "所在城市": {"type": "keyword"},
        "所在区域": {"type": "keyword"},
        "纬度": {"type": "float"},
        "经度": {"type": "float"},
        "分类": {"type": "keyword"},
        "总评分": {"type": "float"},
        "月售": {"type": "integer"},
        "月售原始": {"type": "keyword"},
        "最低起送价": {"type": "float"},
        "配送费": {"type": "float"},
        "预计配送时间": {"type": "integer"},
        "营业时间": {"type": "text"},
        "品牌": {"type": "keyword"},
        "评价总数": {"type": "integer"},
        "味道评分": {"type": "float"},
        "包装评分": {"type": "float"},
        "配送评分": {"type": "float"},
        "配送满意度": {"type": "keyword"},
        "被推荐原因": {
            "type": "text",
            "analyzer": "ik_max_word",      # 索引时使用细粒度分词
            "search_analyzer": "ik_smart"    # 搜索时使用粗粒度分词
        }
    }
}

if es.indices.exists(index=INDEX_NAME):
    print(f"索引 '{INDEX_NAME}' 已存在，删除重建...")
    es.indices.delete(index=INDEX_NAME)

print(f"创建索引 '{INDEX_NAME}'...\n")
es.indices.create(index=INDEX_NAME, mappings=INDEX_MAPPING)

XLS_FILE_PATH = '餐饮外卖商家样本数据.xls'

def extract_number_from_monthly_sales(value):
    """从月售字段提取数字"""
    if pd.isna(value):
        return 0
    value_str = str(value)
    numbers = re.findall(r'\d+', value_str)
    if not numbers:
        return 0
    return int(numbers[0])

def clean_percentage(value):
    """清理百分比字段"""
    if pd.isna(value):
        return "0%"
    return str(value)

try:
    print(f"正在从 '{XLS_FILE_PATH}' 读取数据...")
    df = pd.read_excel(XLS_FILE_PATH, sheet_name='试用数据')
    
    # 数据清洗
    text_fields = ['商家名称', '地址', '营业时间', '被推荐原因', '分类', '品牌']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].astype(str).replace('nan', '').replace('', '无')
    
    keyword_fields = ['所在省份', '所在城市', '所在区域']
    for field in keyword_fields:
        if field in df.columns:
            df[field] = df[field].astype(str).replace('nan', '')
    
    if '月售' in df.columns:
        df['月售原始'] = df['月售'].astype(str)
        df['月售'] = df['月售'].apply(extract_number_from_monthly_sales)
    
    numeric_fields = {
        '总评分': 0.0,
        '纬度': 0.0,
        '经度': 0.0,
        '最低起送价': 0.0,
        '配送费（元）': 0.0,
        '预计配送时间': 0,
        '评价总数': 0,
        '味道评分': 0.0,
        '包装评分': 0.0,
        '配送（骑手）评分': 0.0
    }
    
    for field, default_value in numeric_fields.items():
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(default_value)
    
    df = df.rename(columns={
        '配送费（元）': '配送费',
        '配送（骑手）评分': '配送评分'
    })
    
    if '配送满意度' in df.columns:
        df['配送满意度'] = df['配送满意度'].apply(clean_percentage)
    
    mapping_fields = list(INDEX_MAPPING['properties'].keys())
    available_fields = [f for f in mapping_fields if f in df.columns]
    df = df[available_fields]
    
    print(f"数据读取完成: {df.shape[0]} 条记录\n")
    
except FileNotFoundError:
    print(f"错误：文件 '{XLS_FILE_PATH}' 未找到。", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"读取 Excel 文件时发生错误: {e}", file=sys.stderr)
    sys.exit(1)

def generate_data(dataframe, index_name):
    for idx, row in dataframe.iterrows():
        doc = row.to_dict()
        yield {
            "_index": index_name,
            "_source": doc,
        }

print("开始批量导入数据到 Elasticsearch...")
try:
    success_count = 0
    for ok, result in streaming_bulk(es, generate_data(df, INDEX_NAME), raise_on_error=False):
        if ok:
            success_count += 1
    
    print(f"数据导入完成: 成功导入 {success_count} 条数据\n")
    
except Exception as e:
    print(f"数据导入错误: {e}", file=sys.stderr)
    sys.exit(1)


print("刷新索引，使数据可搜索...")
es.indices.refresh(index=INDEX_NAME)
print("索引刷新完成\n")

print("="*60)
print("查询被推荐原因包含'热情'的商家")
print("="*60)

keyword = "热情"

query = {
    "query": {
        "match": {
            "被推荐原因": keyword
        }
    },
    "_source": ["商家名称", "被推荐原因"],
    "size": 100  # 获取所有匹配结果
}

response = es.search(index=INDEX_NAME, body=query)

print(f"\n共找到 {response['hits']['total']['value']} 家商家的被推荐原因包含'{keyword}'\n")

if response['hits']['total']['value'] > 0:
    for i, hit in enumerate(response['hits']['hits'], 1):
        print(f"{i}. {hit['_source']['商家名称']}")
        print(f"   被推荐原因: {hit['_source']['被推荐原因']}\n")
else:
    print(f"未找到被推荐原因包含'{keyword}'的商家。")

print("="*60)
```


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