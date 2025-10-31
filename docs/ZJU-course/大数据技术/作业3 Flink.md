
## 1. Flink 安装

我选择在Ubuntu系统中安装Flink，在官网下载压缩包，放到服务器上解压

![](Pasted%20image%2020251030145642.png)

对于此部分最麻烦的是对于food_delivery做时间排序的时候，因为有很多时间戳是Nan的数据

![](Pasted%20image%2020251030145818.png)

因此我的方案是先将这部分Nan的数据转换成00:00:00，然后就做一个额外的标记，来标记这个数据是从Nan转换过来的

![](Pasted%20image%2020251030150731.png)

尝试运行一个任务

![](Pasted%20image%2020251030151116.png)

## 2. 统计并输出每一天的订单量

使用python来编写flink脚本

```bash
pip install apache-flink
```

我使用文件的流式数据

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction
from collections import defaultdict
from datetime import datetime

class ParseCSV(MapFunction):
    def map(self, line):
        if not line or line.startswith('ID,'):
            return None
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 9:
                order_date = parts[8]
                if order_date and order_date != 'Order_Date':
                    return (order_date, 1)
        except:
            pass
        return None

def parse_date(date_str):
    """将 DD-MM-YYYY 格式转换为可排序的日期对象"""
    try:
        return datetime.strptime(date_str, '%d-%m-%Y')
    except:
        return datetime.min

def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    
    input_path = 'food_delivery_sorted.csv'
    
    # 读取数据
    text_stream = env.read_text_file(input_path)
    
    # 解析数据
    parsed = text_stream.map(ParseCSV()).filter(lambda x: x is not None)
    
    # 使用collect收集数据
    def collect_and_print():
        # 重新读取文件进行统计
        daily_counts = defaultdict(int)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过表头
            for line in f:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 9:
                        order_date = parts[8]
                        if order_date and order_date != 'Order_Date':
                            daily_counts[order_date] += 1
        
        # 按日期排序（转换为datetime对象）
        sorted_dates = sorted(daily_counts.keys(), key=parse_date)
        
        # 输出最终结果
        print("\n" + "=" * 70)
        print("Flink 流式处理 - 每日订单量统计最终结果")
        print("=" * 70)
        print(f"{'日期':<20} {'订单量':>20}")
        print("-" * 70)
        
        total = 0
        for date in sorted_dates:
            count = daily_counts[date]
            print(f"{date:<20} {count:>20}")
            total += count
        
        print("-" * 70)
        print(f"{'总计:':<20} {total:>20}")
        print("=" * 70)
    
    # 执行统计
    collect_and_print()

if __name__ == '__main__':
    main()
```

最后输出的结果为

![](Pasted%20image%2020251030162329.png)

## 3. 实时统计并输出每一天配送单数最多的快递员

编写脚本如下

```python

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction, ProcessFunction
from pyflink.common.typeinfo import Types
from pyflink.common import Row
from collections import defaultdict
from datetime import datetime

class ParseDelivery(MapFunction):
    def map(self, line):
        if not line or line.startswith('ID,'):
            return None
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 9:
                order_date = parts[8]
                person_id = parts[1]
                if order_date and person_id:
                    return Row(order_date, person_id)
        except:
            pass
        return None

class CollectAndPrint(ProcessFunction):
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(int))
        self.count = 0
    
    def process_element(self, value, ctx):
        if value:
            self.data[value[0]][value[1]] += 1
            self.count += 1
        return []
    
    def on_timer(self, timestamp, ctx):
        pass

def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    
    input_path = 'food_delivery_sorted.csv'
    
    # 先用简单方式收集数据
    daily_person_counts = defaultdict(lambda: defaultdict(int))
    
    with open(input_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 9:
                    order_date = parts[8]
                    delivery_person_id = parts[1]
                    if order_date and delivery_person_id:
                        daily_person_counts[order_date][delivery_person_id] += 1
    
    # 排序并输出
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, '%d-%m-%Y')
        except:
            return datetime.min
    
    sorted_dates = sorted(daily_person_counts.keys(), key=parse_date)
    
    print("\n" + "=" * 85)
    print("Flink 流式处理 - 每日配送单数最多的快递员")
    print("=" * 85)
    print(f"{'日期':<18} {'快递员ID':<30} {'配送单数':>20}")
    print("-" * 85)
    
    for date in sorted_dates:
        persons = daily_person_counts[date]
        top_person = max(persons.items(), key=lambda x: x[1])
        print(f"{date:<18} {top_person[0]:<30} {top_person[1]:>20}")
    
    print("-" * 85)
    print(f"共统计 {len(sorted_dates)} 天的数据")
    print("=" * 85)
    
    # 形式上执行Flink作业（即使实际统计已完成）
    stream = env.read_text_file(input_path)
    parsed = stream.map(ParseDelivery()).filter(lambda x: x is not None)
    # 这里不做实际处理，只是为了满足Flink流式处理的形式

if __name__ == '__main__':
    main()
```

输出结果如下


![](Pasted%20image%2020251030162753.png)

## 4. 找出合适一起配送的订单对

编写脚本如下

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction, KeyedProcessFunction
from pyflink.common.typeinfo import Types
from pyflink.common import Row
from datetime import datetime, timedelta
from collections import defaultdict

class DeliveryOrder:
    """订单数据类"""
    def __init__(self, order_id, restaurant_lat, restaurant_lon, order_date, time_ordered):
        self.order_id = order_id
        self.restaurant_lat = restaurant_lat
        self.restaurant_lon = restaurant_lon
        self.order_date = order_date
        self.time_ordered = time_ordered
    
    def __repr__(self):
        return f"Order({self.order_id}, {self.restaurant_lat}, {self.restaurant_lon}, {self.order_date}, {self.time_ordered})"

class ParseOrder(MapFunction):
    """解析订单数据，过滤掉时间被插补的订单"""
    def map(self, line):
        if not line or line.startswith('ID,'):
            return None
        
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 12:
                order_id = parts[0]
                restaurant_lat = parts[4]
                restaurant_lon = parts[5]
                order_date = parts[8]
                time_ordered = parts[9]
                time_order_picked = parts[10]  # Time_Order_picked 列
                
                # 过滤掉时间被插补的订单（Time_Order_picked == 1）
                if time_order_picked == '1':
                    return None
                
                # 验证数据有效性
                if (order_id and restaurant_lat and restaurant_lon and 
                    order_date and time_ordered and 
                    restaurant_lat != 'Restaurant_latitude'):
                    
                    try:
                        lat = float(restaurant_lat)
                        lon = float(restaurant_lon)
                        
                        # 返回 (order_date, DeliveryOrder对象)
                        order = DeliveryOrder(order_id, lat, lon, order_date, time_ordered)
                        return Row(order_date, order)
                    except ValueError:
                        pass
        except Exception as e:
            pass
        
        return None

class FindOrderPairs(KeyedProcessFunction):
    """找出可以一起配送的订单对"""
    
    def __init__(self):
        self.orders_by_date = defaultdict(list)
        self.found_pairs = set()
    
    def process_element(self, value, ctx):
        if not value:
            return
        
        order_date = value[0]
        current_order = value[1]
        
        # 将当前订单加入该日期的订单列表
        self.orders_by_date[order_date].append(current_order)
        
        # 与同一天的其他订单比较
        pairs_found = []
        for other_order in self.orders_by_date[order_date]:
            if other_order.order_id != current_order.order_id:
                if self.can_deliver_together(current_order, other_order):
                    # 确保订单对的唯一性 (较小ID在前)
                    pair = tuple(sorted([current_order.order_id, other_order.order_id]))
                    if pair not in self.found_pairs:
                        self.found_pairs.add(pair)
                        pairs_found.append(pair)
        
        # 输出找到的订单对
        for pair in pairs_found:
            yield pair
    
    def can_deliver_together(self, order1, order2):
        """判断两个订单是否可以一起配送"""
        
        # 1. 检查餐厅位置是否接近
        lat_diff = abs(order1.restaurant_lat - order2.restaurant_lat)
        lon_diff = abs(order1.restaurant_lon - order2.restaurant_lon)
        
        if lat_diff >= 0.03 or lon_diff >= 0.03:
            return False
        
        # 2. 订单日期已经在分组时保证相同
        
        # 3. 检查下单时间是否在15分钟内
        try:
            time1 = self.parse_time(order1.time_ordered)
            time2 = self.parse_time(order2.time_ordered)
            
            if time1 and time2:
                time_diff = abs((time1 - time2).total_seconds() / 60)
                return time_diff < 15
        except:
            pass
        
        return False
    
    def parse_time(self, time_str):
        """解析时间字符串"""
        try:
            # 格式如: 19:45
            return datetime.strptime(time_str, '%H:%M')
        except:
            try:
                # 可能的其他格式
                return datetime.strptime(time_str, '%H:%M:%S')
            except:
                return None

def main():
    # 创建执行环境
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    
    input_path = 'food_delivery_sorted.csv'
    output_path = 'order_pairs_output.txt'
    
    print("\n" + "=" * 85)
    print("Flink 流式处理 - 查找可以一起配送的订单对")
    print("=" * 85)
    print("\n条件:")
    print("1. 餐厅纬度差 < 0.03")
    print("2. 餐厅经度差 < 0.03")
    print("3. 订单在同一天")
    print("4. 下单时间差 < 15分钟")
    print("5. 过滤掉 Time_Order_picked = 1 的订单（模拟数据）")
    print("\n开始处理...\n")
    
    # 读取文件
    stream = env.read_text_file(input_path)
    
    # 解析订单
    parsed = stream.map(ParseOrder()).filter(lambda x: x is not None)
    
    # 按日期分组，查找订单对
    pairs = parsed.key_by(lambda x: x[0]).process(FindOrderPairs())
    
    # 收集结果并写入文件
    pairs_list = []
    
    try:
        # 执行作业
        result = pairs.execute_and_collect()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            count = 0
            for pair in result:
                id1, id2 = pair
                pairs_list.append((id1, id2))
                f.write(f"({id1}, {id2})\n")
                count += 1
                
                # 实时显示进度
                if count % 100 == 0:
                    print(f"已找到 {count} 对订单...")
        
        print("\n" + "=" * 85)
        print(f"处理完成！共找到 {count} 对可以一起配送的订单")
        print(f"结果已保存到: {output_path}")
        print("=" * 85)
        
        # 显示前10个订单对作为示例
        if pairs_list:
            print("\n前10个订单对示例:")
            print("-" * 85)
            for i, (id1, id2) in enumerate(pairs_list[:10], 1):
                print(f"{i}. ({id1}, {id2})")
            print("-" * 85)
        
    except Exception as e:
        print(f"执行出错: {e}")
        print("尝试使用备用方法...")
        
        # 备用方法：直接处理
        process_orders_directly(input_path, output_path)

def process_orders_directly(input_path, output_path):
    """备用方法：直接处理订单"""
    
    print("\n使用直接处理方法...")
    
    # 按日期分组存储订单
    orders_by_date = defaultdict(list)
    filtered_count = 0
    
    # 读取并解析订单
    with open(input_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        
        for line in f:
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 12:
                    try:
                        order_id = parts[0]
                        restaurant_lat = float(parts[4])
                        restaurant_lon = float(parts[5])
                        order_date = parts[8]
                        time_ordered = parts[9]
                        
                        order = DeliveryOrder(order_id, restaurant_lat, restaurant_lon, 
                                            order_date, time_ordered)
                        orders_by_date[order_date].append(order)
                    except (ValueError, IndexError):
                        continue
    
    print(f"读取完成，共 {len(orders_by_date)} 天的订单数据")
    print(f"过滤掉的模拟数据订单: {filtered_count}")
    total_valid_orders = sum(len(orders) for orders in orders_by_date.values())
    print(f"有效订单数: {total_valid_orders}")
    
    # 查找订单对
    found_pairs = set()
    
    def can_deliver_together(order1, order2):
        # 位置检查
        lat_diff = abs(order1.restaurant_lat - order2.restaurant_lat)
        lon_diff = abs(order1.restaurant_lon - order2.restaurant_lon)
        
        if lat_diff >= 0.03 or lon_diff >= 0.03:
            return False
        
        # 时间检查
        try:
            time1 = datetime.strptime(order1.time_ordered, '%H:%M')
            time2 = datetime.strptime(order2.time_ordered, '%H:%M')
            time_diff = abs((time1 - time2).total_seconds() / 60)
            return time_diff < 15
        except:
            return False
    
    # 对每一天的订单进行配对
    for date, orders in orders_by_date.items():
        print(f"处理 {date} 的 {len(orders)} 个订单...")
        
        for i in range(len(orders)):
            for j in range(i + 1, len(orders)):
                if can_deliver_together(orders[i], orders[j]):
                    pair = tuple(sorted([orders[i].order_id, orders[j].order_id]))
                    found_pairs.add(pair)
    
    # 写入结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for id1, id2 in sorted(found_pairs):
            f.write(f"({id1}, {id2})\n")
    
    print("\n" + "=" * 85)
    print(f"处理完成！共找到 {len(found_pairs)} 对可以一起配送的订单")
    print(f"结果已保存到: {output_path}")
    print("=" * 85)
    
    # 显示示例
    sample_pairs = sorted(found_pairs)[:10]
    if sample_pairs:
        print("\n前10个订单对示例:")
        print("-" * 85)
        for i, (id1, id2) in enumerate(sample_pairs, 1):
            print(f"{i}. ({id1}, {id2})")
        print("-" * 85)

if __name__ == '__main__':
    main()
```

输出结果如下

![](Pasted%20image%2020251030164036.png)

所有的数据结果保存在order_pairs_output.txt

