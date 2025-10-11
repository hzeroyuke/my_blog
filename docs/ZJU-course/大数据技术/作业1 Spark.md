
## 1. Spark 安装

我选择在Ubuntu（Linux）系统中安装Spark，首先Wget安装包

![[Pasted image 20251004200311.png]]

随后解压，并且设置环境变量

![[Pasted image 20251004200911.png]]

随后启动成功，屏幕截图如下

![[Pasted image 20251004200840.png]]


## 2. Spark WordCount

首先我们使用一行Bash命令来处理这个CSV文件，提取出需要的一行，保存在一个txt文件中

``` bash
tail -n +2 food_delivery.csv | cut -d ',' -f 9 > order_dates.txt
```

随后使用如下命令，启动 wordcount 程序

``` bash
$SPARK_HOME/bin/spark-submit \ $SPARK_HOME/examples/src/main/python/wordcount.py \ order_dates.txt
```

最终结果如下

![[Pasted image 20251004201612.png]]


## 3. 查找年龄最大人员ID

这个任务，我选择用Python来进行Spark编程，首先我们需要下载PySpark包，直接使用pip install

代码如下

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, trim

spark = SparkSession.builder.appName("FindOldestDeliveryPerson").getOrCreate()

file_path = "food_delivery.csv" # 需要处理的文件

df = spark.read.csv(file_path, header=True, inferSchema=True) # 读取csv

# 清洗数据
df_clean = (
    df
    .withColumn("Delivery_person_Age", trim(col("Delivery_person_Age")))
    .dropna(subset=["Delivery_person_Age"])
    .withColumn("Age_Int", col("Delivery_person_Age").cast("float").try_cast("int"))
    .dropna(subset=["Age_Int"])
)

max_age = df_clean.agg(max("Age_Int")).collect()[0][0] # 查找最大年龄

# 找出年龄等于最大年龄的配送人员ID
oldest_drivers_df = (
    df_clean.filter(col("Age_Int") == max_age)
    .select("Delivery_person_ID", "Age_Int")
    .distinct()
    .orderBy(col("Delivery_person_ID"))

)
# 打印结果
print("\n" + "="*50)
print(f"任务结果：年龄最大的配送人员ID（最大年龄：{max_age}）")
print("="*50)
total_rows = oldest_drivers_df.count()
oldest_drivers_df.show(total_rows, truncate=False)
print("="*50 + "\n")

spark.stop()
```

用spark-submit来运行上述程序，代码运行结果如下

![](Pasted%20image%2020251006192846.png)


![](Pasted%20image%2020251006192912.png)

## 4. 统计平均的外卖时间

编写代码如下

``` python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, to_date, year, month, split
spark = SparkSession.builder.appName("AverageDeliveryTimeMarch2022").getOrCreate()

file_path = "food_delivery.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True) # 定义处理文件

# 数据清洗与转换
df_transformed = df.withColumn(
    "OrderDate",
    to_date(col("Order_Date"), "dd-MM-yyyy")
).withColumn(
    "TimeInMinutes",
    split(col("Time_taken(min)"), ' ')[1].cast("int")
)

# 找出2022年3月份的订单
march_2022_orders_df = df_transformed.filter(
    (year(col("OrderDate")) == 2022) & (month(col("OrderDate")) == 3)
)

# 计算平均配送时间
avg_time_df = march_2022_orders_df.dropna(subset=["TimeInMinutes"]).agg(
    avg("TimeInMinutes").alias("average_delivery_time")
)

# 提取并打印结果
average_time = avg_time_df.collect()[0]['average_delivery_time']

print("\n" + "="*60)
print(f"任务结果：2022年3月份的外卖订单平均配送时间为: {average_time:.2f} 分钟")
print("="*60 + "\n")
spark.stop()
```

实验结果截图如下

![[Pasted image 20251004213844.png]]


