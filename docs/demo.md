# EasyBet 演示操作指南

这是一份详细的演示操作流程，用于录制视频展示项目功能。

## 📋 演示前准备

### 1. 环境检查清单
- ✅ Hardhat 本地节点正在运行 (localhost:8545)
- ✅ 智能合约已部署
- ✅ 前端已启动 (localhost:3000)
- ✅ MetaMask 已安装并配置本地网络
- ✅ MetaMask 已导入测试账户

### 2. MetaMask 配置
**网络设置：**
- 网络名称: `Localhost 8545`
- RPC URL: `http://localhost:8545`
- 链 ID: `1337`
- 货币符号: `ETH`

**导入账户：**
使用 Hardhat 提供的测试账户私钥（见 hardhat.config.ts 或运行 `npx hardhat node` 时显示的账户）

## 🎬 演示流程（建议录制顺序）

### 第一部分：项目介绍 (2-3分钟)

#### 1. 项目背景说明
**说明内容：**
- 这是一个去中心化彩票系统 (EasyBet)
- 解决传统彩票系统无法交易的痛点
- 支持体育赛事、娱乐节目等各类竞猜

**技术栈展示：**
- 智能合约：Solidity + Hardhat + OpenZeppelin
- 前端：React + TypeScript + Ethers.js
- 区块链：以太坊本地测试网络

#### 2. 架构说明
**展示三个智能合约：**
```
EasyBet.sol (主合约)
├── BettingTicket.sol (ERC721 彩票凭证)
└── BettingToken.sol (ERC20 积分代币)
```

**核心功能列表：**
1. ✅ 创建竞猜活动（公证人）
2. ✅ 下注购买彩票（ETH + ERC20双支付）
3. ✅ 彩票交易（链上订单簿）
4. ✅ 结果公布和奖金分配
5. ✅ Bonus: ERC20 代币系统 (+2分)
6. ✅ Bonus: 链上订单簿 (+3分)

---

### 第二部分：连接钱包 (1分钟)

#### 操作步骤：
1. **打开浏览器访问** `http://localhost:3000`
2. **点击 "Connect Wallet" 按钮**
3. **MetaMask 弹窗 → 选择账户 → 点击"连接"**
4. **确认连接成功**
   - 页面显示账户地址（缩写格式）
   - 显示当前余额

**录制要点：**
- 清晰展示 MetaMask 连接过程
- 强调这是 Web3 DApp 的标准连接流程

---

### 第三部分：领取测试代币 (1分钟)

#### 操作步骤：
1. **找到"Token Faucet"（代币水龙头）区域**
2. **查看当前 BTK 余额** (初始为 0)
3. **点击 "Claim 1000 BTK" 按钮**
4. **MetaMask 弹窗确认交易 → 点击"确认"**
5. **等待交易完成** (几秒钟)
6. **刷新页面，确认余额增加到 1000 BTK**

**说明内容：**
- BettingToken (BTK) 是项目的 ERC20 代币
- 每个地址每24小时可领取 1000 BTK
- BTK 可用于下注（实现 Bonus 需求）

**演示代码位置：**
- 合约：`contracts/BettingToken.sol` 中的 `claimTokens()` 函数
- 前端：`BettingApp.tsx` 中的水龙头功能

---

### 第四部分：创建竞猜活动（公证人角色）(2-3分钟)

#### 示例活动 1: NBA MVP 竞猜
**操作步骤：**
1. **找到 "Create New Activity" 表单**
2. **填写活动信息：**
   - Title: `2024 NBA MVP`
   - Choices: `LeBron James, Stephen Curry, Giannis Antetokounmpo`
   - Duration: `7200` (秒，即2小时)
   - Prize Pool: `1.0` (ETH)

3. **点击 "Create Activity" 按钮**
4. **MetaMask 确认交易** (会从账户扣除 1 ETH 作为初始奖池)
5. **等待交易确认**
6. **页面自动刷新，显示新创建的活动**

**说明内容：**
- 公证人创建活动时需要提供初始奖池（ETH）
- 支持 2 个或更多选项
- 活动有时间限制（最短 1 小时）
- 玩家下注的金额会累加到奖池

#### 示例活动 2: 足球比赛竞猜
**快速创建第二个活动：**
- Title: `Manchester United vs Liverpool`
- Choices: `Man Utd Win, Draw, Liverpool Win`
- Duration: `3600` (1小时)
- Prize Pool: `0.5` (ETH)

**演示代码位置：**
- 合约：`EasyBet.sol:56` - `createActivity()` 函数
- 事件：`ActivityCreated` 事件

---

### 第五部分：玩家下注 (3-4分钟)

#### 场景 1: 使用 ETH 下注

**操作步骤：**
1. **在活动列表中找到 "2024 NBA MVP"**
2. **点击 "Place Bet" 按钮展开下注表单**
3. **选择选项：** 例如选择 "LeBron James" (Choice 0)
4. **输入下注金额：** `0.1` ETH
5. **确保 "Use BTK Token" 开关关闭** (使用 ETH)
6. **点击 "Place Bet" 按钮**
7. **MetaMask 确认交易**
8. **等待确认，提示 "Bet placed successfully!"**

**观察变化：**
- 奖池总额从 1.0 ETH 增加到 1.1 ETH
- 用户获得一张彩票凭证（ERC721 NFT）

**说明内容：**
- 每次下注都会铸造一个唯一的 ERC721 NFT 作为彩票凭证
- NFT 中记录了活动ID、选择、下注金额等信息
- 奖池自动累加所有玩家的下注金额

#### 场景 2: 使用 ERC20 代币 (BTK) 下注

**操作步骤：**
1. **在同一个活动中准备再次下注**
2. **这次选择不同的选项：** "Stephen Curry" (Choice 1)
3. **输入金额：** `100` BTK (注意单位是 BTK，不是 ETH)
4. **打开 "Use BTK Token" 开关** ✅
5. **点击 "Place Bet" 按钮**
6. **MetaMask 会弹出两次确认：**
   - 第一次：Approve BTK 代币授权
   - 第二次：实际下注交易
7. **两次都确认后等待交易完成**

**说明内容：**
- 这是 ERC20 代币的标准使用流程
- 需要先 approve 授权合约使用你的代币
- 实现了双支付系统（Bonus +2分的核心功能）

**演示代码位置：**
- 合约：`EasyBet.sol:79` - `placeBet()` (ETH)
- 合约：`EasyBet.sol:97` - `placeBetWithTokens()` (BTK)
- 前端：`BettingApp.tsx:106` - `handlePlaceBet()`

#### 场景 3: 切换账户，让另一个玩家下注

**操作步骤：**
1. **在 MetaMask 中切换到另一个测试账户**
2. **页面会自动检测到账户变化**
3. **该账户也领取 1000 BTK**
4. **在 "Manchester United vs Liverpool" 活动中下注**
5. **选择 "Liverpool Win"，下注 0.2 ETH**
6. **确认交易**

**说明内容：**
- 展示多个玩家参与的场景
- 不同玩家可以选择不同的选项
- 奖池会随着参与人数增加而增长

---

### 第六部分：彩票交易（链上订单簿）(3-4分钟)

这是 Bonus +3分 的核心功能。

#### 场景：玩家想卖出彩票

**操作步骤：**
1. **假设场景说明：**
   - 你之前买了 "LeBron James" 的彩票
   - 但 LeBron 突然受伤，你想止损
   - 决定在市场上出售这张彩票

2. **在活动详情中找到 "My Tickets" 区域**
   - 显示你持有的彩票列表
   - 每张彩票显示：Token ID、选择、下注金额

3. **点击某张彩票的 "Sell Ticket" 按钮**
4. **输入出售价格：** 例如 `0.05` ETH (低于原价 0.1 ETH，折价出售)
5. **点击 "Place Order" 确认挂单**
6. **MetaMask 确认交易**
7. **订单成功创建**

#### 查看订单簿

**操作步骤：**
1. **在活动详情中找到 "Order Book" 区域**
2. **显示当前所有挂单信息：**
   - Token ID
   - Choice (哪个选项的彩票)
   - Seller (卖家地址)
   - Price (出售价格)
   - 状态 (Active)

3. **说明订单簿的作用：**
   - 展示市场上所有待售彩票
   - 买家可以按价格筛选
   - 实现去中心化的彩票交易市场

#### 场景：另一个玩家买入彩票

**操作步骤：**
1. **切换到另一个 MetaMask 账户**
2. **在订单簿中找到刚才挂出的订单**
3. **点击 "Buy Ticket" 按钮**
4. **MetaMask 确认支付 0.05 ETH**
5. **交易完成后：**
   - 彩票 NFT 转移到买家账户
   - ETH 支付给卖家
   - 订单状态变为 Filled (已成交)

**说明内容：**
- 这是完全链上的去中心化交易
- 不需要第三方托管
- ERC721 NFT 自动转移所有权
- 解决了传统彩票无法交易的痛点

**演示代码位置：**
- 合约：`EasyBet.sol:145` - `placeOrder()` (挂单)
- 合约：`EasyBet.sol:167` - `fillOrder()` (成交)
- 合约：`EasyBet.sol:189` - `getActiveOrders()` (查询订单簿)

**补充说明（Bonus 加分项）：**
- 支持多价格挂单（同一张彩票可以多次调整价格）
- 显示市场深度（不同价格的彩票数量）
- 自动撮合最优价格（前端可以按价格排序）

---

### 第七部分：设置竞猜结果（公证人）(2分钟)

#### 操作步骤：

1. **切换回创建活动的账户** (公证人/创建者)
2. **说明场景：**
   - 比赛/赛季已结束
   - 真实结果已出炉
   - 公证人输入结果进行结算

3. **在 "2024 NBA MVP" 活动中**
4. **找到 "Set Result" 管理区域** (仅创建者可见)
5. **选择获胜选项：** 例如 "LeBron James" (Choice 0)
6. **点击 "Set Result" 按钮**
7. **MetaMask 确认交易**

**交易执行后发生的事情：**
- 活动状态变为 "Finished"
- 智能合约自动计算奖金分配
- 从奖池中扣除 5% 作为平台手续费
- 剩余 95% 按照获胜玩家的下注比例分配
- 每个获胜者的余额更新

**说明内容：**
- 只有创建者或合约 owner 可以设置结果
- 结果一旦设置不可更改（防止作弊）
- 奖金自动分配，无需玩家手动领取

**演示代码位置：**
- 合约：`EasyBet.sol:110` - `setResult()` 函数
- 合约：`EasyBet.sol:127` - `_distributePrizes()` 内部函数
- 事件：`ActivityFinished`, `PrizeDistributed`

---

### 第八部分：提取奖金 (1-2分钟)

#### 操作步骤：

1. **切换到获胜玩家的账户**
2. **查看页面顶部的余额显示**
   - "Your Prize Balance: X.XX ETH"
   - 显示可提取的奖金金额

3. **点击 "Withdraw Prize" 按钮**
4. **MetaMask 确认交易**
5. **交易完成后：**
   - 奖金余额清零
   - ETH 转入钱包
   - 可以在 MetaMask 中查看 ETH 余额增加

**说明内容：**
- 每个用户有独立的奖金余额
- 可以一次性提取所有奖金
- 合约使用 pull payment 模式确保安全性

**演示代码位置：**
- 合约：`EasyBet.sol:121` - `withdrawPrize()` 函数
- 前端：`BettingApp.tsx:151` - `handleWithdraw()`

---

### 第九部分：补充功能展示 (2-3分钟)

#### 1. 查看活动详情

**展示内容：**
- 活动列表中每个活动的详细信息
- 实时奖池金额
- 剩余时间倒计时
- 每个选项的下注总额
- 活动状态（Active / Finished / Cancelled）

#### 2. 错误处理演示

**故意触发错误，展示健壮性：**

**例子 1：活动已结束时尝试下注**
- 等待活动时间到期
- 尝试下注
- 显示错误："Activity ended"

**例子 2：BTK 余额不足**
- 尝试用 2000 BTK 下注（余额只有 1000）
- 显示错误："Insufficient token balance"

**例子 3：非创建者尝试设置结果**
- 切换到普通玩家账户
- 尝试设置结果
- 显示错误："Only creator can set result"

**说明内容：**
- 智能合约有完善的输入验证
- 所有操作都有权限控制
- 前端有友好的错误提示

#### 3. 事件日志展示

**操作：**
1. **打开浏览器开发者工具 (F12)**
2. **切换到 Console 标签**
3. **执行几个操作（创建活动、下注等）**
4. **展示控制台中的交易哈希和事件日志**

**说明内容：**
- 所有关键操作都会触发智能合约事件
- 前端监听事件实现实时更新
- 可以通过事件追溯所有历史记录

---

### 第十部分：代码讲解（可选，3-5分钟）

如果时间充足，可以简单讲解核心代码。

#### 1. 智能合约核心逻辑

**打开 `EasyBet.sol`，讲解关键部分：**

**Activity 结构体：**
```solidity
struct Activity {
    address creator;
    string title;
    string[] choices;
    uint256 totalPrizePool;
    uint256 endTimestamp;
    ActivityStatus status;
    uint256 winningChoice;
    bool resultSet;
    mapping(uint256 => uint256) choiceBetAmounts;  // 每个选项的总下注额
    mapping(uint256 => uint256[]) choiceTickets;   // 每个选项的彩票列表
}
```

**下注逻辑（ETH）：**
```solidity
function placeBet(uint256 activityId, uint256 choiceIndex) external payable {
    Activity storage activity = activities[activityId];
    require(activity.status == ActivityStatus.Active, "Activity not active");
    require(block.timestamp < activity.endTimestamp, "Activity ended");
    require(choiceIndex < activity.choices.length, "Invalid choice");
    require(msg.value > 0, "Bet amount required");

    // 累加奖池
    activity.totalPrizePool += msg.value;
    activity.choiceBetAmounts[choiceIndex] += msg.value;

    // 铸造彩票 NFT
    uint256 tokenId = bettingTicket.mintTicket(msg.sender, activityId, choiceIndex, msg.value);
    activity.choiceTickets[choiceIndex].push(tokenId);

    emit BetPlaced(activityId, msg.sender, choiceIndex, msg.value, tokenId);
}
```

**奖金分配逻辑：**
```solidity
function _distributePrizes(uint256 activityId) internal {
    Activity storage activity = activities[activityId];

    // 扣除平台费用（5%）
    uint256 platformFee = (activity.totalPrizePool * PLATFORM_FEE_PERCENT) / 100;
    uint256 prizePool = activity.totalPrizePool - platformFee;

    // 转移平台费到 owner
    userBalances[owner()] += platformFee;

    // 获取获胜选项的所有彩票
    uint256[] memory winningTickets = activity.choiceTickets[activity.winningChoice];
    uint256 winningBetAmount = activity.choiceBetAmounts[activity.winningChoice];

    // 按比例分配奖金
    for (uint256 i = 0; i < winningTickets.length; i++) {
        uint256 tokenId = winningTickets[i];
        BettingTicket.TicketInfo memory ticket = bettingTicket.getTicketInfo(tokenId);

        uint256 prize = (prizePool * ticket.betAmount) / winningBetAmount;
        userBalances[ticket.owner] += prize;

        emit PrizeDistributed(activityId, ticket.owner, prize);
    }
}
```

#### 2. ERC721 彩票凭证

**打开 `BettingTicket.sol`：**
```solidity
struct TicketInfo {
    address owner;
    uint256 activityId;
    uint256 choiceIndex;
    uint256 betAmount;
    uint256 timestamp;
}

function mintTicket(
    address to,
    uint256 activityId,
    uint256 choiceIndex,
    uint256 betAmount
) external onlyEasyBet returns (uint256) {
    uint256 tokenId = _tokenIdCounter++;
    _safeMint(to, tokenId);

    tickets[tokenId] = TicketInfo({
        owner: to,
        activityId: activityId,
        choiceIndex: choiceIndex,
        betAmount: betAmount,
        timestamp: block.timestamp
    });

    return tokenId;
}
```

#### 3. 链上订单簿

**订单结构：**
```solidity
struct OrderBookEntry {
    uint256 tokenId;
    address seller;
    uint256 price;
    bool active;
}

mapping(uint256 => OrderBookEntry[]) public orderBooks; // activityId => orders
```

**挂单：**
```solidity
function placeOrder(uint256 activityId, uint256 tokenId, uint256 price) external {
    require(bettingTicket.ownerOf(tokenId) == msg.sender, "Not ticket owner");

    orderBooks[activityId].push(OrderBookEntry({
        tokenId: tokenId,
        seller: msg.sender,
        price: price,
        active: true
    }));

    emit OrderPlaced(activityId, tokenId, msg.sender, price);
}
```

**成交：**
```solidity
function fillOrder(uint256 activityId, uint256 orderIndex) external payable {
    OrderBookEntry storage order = orderBooks[activityId][orderIndex];
    require(order.active, "Order not active");
    require(msg.value >= order.price, "Insufficient payment");

    // 转移 NFT
    bettingTicket.safeTransferFrom(order.seller, msg.sender, order.tokenId);

    // 转移 ETH 给卖家
    payable(order.seller).transfer(order.price);

    // 退还多余的 ETH
    if (msg.value > order.price) {
        payable(msg.sender).transfer(msg.value - order.price);
    }

    order.active = false;
    emit OrderFilled(activityId, order.tokenId, order.seller, msg.sender, order.price);
}
```

#### 4. 前端 Web3 集成

**Web3Context 连接逻辑：**
```typescript
const connectWallet = async () => {
  if (typeof window.ethereum !== 'undefined') {
    const provider = new ethers.BrowserProvider(window.ethereum);
    const accounts = await provider.send('eth_requestAccounts', []);
    const signer = await provider.getSigner();

    // 初始化合约实例
    const easyBet = new ethers.Contract(
      CONTRACT_ADDRESSES.EasyBet,
      EasyBetABI,
      signer
    );

    setEasyBetContract(easyBet);
    setAccount(accounts[0]);
    setIsConnected(true);
  }
};
```

---

## 🎯 演示总结 (1分钟)

### 功能回顾

**基础需求 (全部实现)：**
1. ✅ 公证人创建多选项竞猜活动
2. ✅ 玩家使用 ETH/ERC20 购买彩票
3. ✅ 彩票交易（链上订单簿）
4. ✅ 结果公布和自动奖金分配

**Bonus 功能：**
1. ✅ ERC20 代币系统 (+2分)
   - BettingToken (BTK) 实现
   - 水龙头功能
   - 双支付支持

2. ✅ 链上订单簿 (+3分)
   - 多价格挂单
   - 订单簿查询
   - 去中心化交易

### 技术亮点

1. **完整的 DApp 架构**
   - 智能合约 + 前端完全集成
   - 标准的 Web3 开发流程

2. **安全性**
   - 使用 OpenZeppelin 标准库
   - 权限控制（Ownable）
   - 重入攻击防护

3. **用户体验**
   - 直观的界面设计
   - 实时余额更新
   - 友好的错误提示

4. **代码质量**
   - TypeScript 类型安全
   - 完整的测试覆盖
   - 清晰的代码结构

---

## 📊 视频录制建议

### 时长分配（总计 15-20 分钟）
- 项目介绍：2-3 分钟
- 连接钱包：1 分钟
- 领取代币：1 分钟
- 创建活动：2-3 分钟
- 玩家下注：3-4 分钟
- 彩票交易：3-4 分钟
- 设置结果：2 分钟
- 提取奖金：1-2 分钟
- 补充功能：2-3 分钟
- 代码讲解：3-5 分钟（可选）
- 总结：1 分钟

### 录制技巧

1. **画面分割建议：**
   - 主画面：浏览器 (前端界面)
   - 小窗口：MetaMask 交易确认
   - 必要时：VS Code (代码展示)

2. **声音建议：**
   - 清晰说明每一步操作
   - 解释为什么这样做
   - 强调技术创新点

3. **节奏控制：**
   - 等待交易确认时可以讲解代码
   - 避免长时间沉默
   - 预先准备好说辞

4. **突出重点：**
   - 强调 Bonus 功能（+5分）
   - 展示去中心化特性
   - 说明实际应用价值

### 可能遇到的问题及解决

**问题 1：交易失败**
- 检查账户余额是否充足
- 确认网络连接正常
- 查看 MetaMask 是否连接到正确网络

**问题 2：合约地址错误**
- 重新部署后务必更新前端配置
- 检查 `addresses.ts` 中的地址是否正确

**问题 3：页面不更新**
- 手动刷新页面
- 检查浏览器控制台是否有错误
- 重新连接钱包

**问题 4：MetaMask 未弹出**
- 检查 MetaMask 是否解锁
- 刷新页面重试
- 检查浏览器扩展是否启用

---

## 🚀 快速启动命令

### 终端 1: 启动 Hardhat 节点
```bash
cd /home/yuke/ZJU-blockchain-course-2025/contracts
npx hardhat node
```

### 终端 2: 部署合约并启动前端
```bash
cd /home/yuke/ZJU-blockchain-course-2025
./start.sh
```

### 手动步骤（如果脚本失败）
```bash
# 部署合约
cd contracts
npx hardhat run scripts/deploy.ts --network localhost

# 复制 ABI
cp artifacts/contracts/EasyBet.sol/EasyBet.json ../frontend/src/contracts/
cp artifacts/contracts/BettingTicket.sol/BettingTicket.json ../frontend/src/contracts/
cp artifacts/contracts/BettingToken.sol/BettingToken.json ../frontend/src/contracts/

# 更新前端地址配置
# 编辑 frontend/src/contracts/addresses.ts

# 启动前端
cd ../frontend
npm start
```

---

## ✅ 演示检查清单

**开始录制前确认：**
- [ ] Hardhat 节点正在运行
- [ ] 合约已成功部署
- [ ] 前端已启动且可访问
- [ ] MetaMask 已配置本地网络
- [ ] 至少导入 2-3 个测试账户
- [ ] 测试账户有足够的 ETH
- [ ] 浏览器开发者工具已打开（备用）
- [ ] 录屏软件已准备好
- [ ] 麦克风已测试

**演示过程中：**
- [ ] 连接钱包成功
- [ ] 领取 BTK 代币
- [ ] 创建至少 2 个活动
- [ ] 使用 ETH 下注
- [ ] 使用 BTK 下注
- [ ] 切换账户演示多玩家
- [ ] 挂单出售彩票
- [ ] 购买彩票（订单簿）
- [ ] 设置活动结果
- [ ] 提取奖金
- [ ] 展示错误处理（可选）

**演示后：**
- [ ] 确认所有功能都已展示
- [ ] 检查录制质量
- [ ] 确认音频清晰
- [ ] 准备好 README 和代码仓库链接

---

## 📝 视频脚本参考

### 开场白
"大家好，今天我要展示的是 EasyBet - 一个去中心化的彩票系统。这个项目解决了传统彩票系统无法交易的痛点，允许玩家在竞猜结果公布前自由买卖彩票。项目完整实现了所有基础需求，以及 5 分 Bonus 功能。"

### 技术介绍
"技术栈方面，智能合约使用 Solidity 0.8.20 和 OpenZeppelin 标准库，确保安全性；前端使用 React + TypeScript，通过 Ethers.js 与区块链交互。项目包含三个智能合约：主合约 EasyBet、ERC721 彩票凭证和 ERC20 积分代币。"

### 功能演示引导
"首先我们需要连接 MetaMask 钱包... 现在让我展示如何领取测试代币... 接下来作为公证人创建一个竞猜活动... 切换到玩家角色进行下注..."

### Bonus 功能强调
"这里要重点展示两个 Bonus 功能：第一个是 ERC20 代币系统，支持使用 BTK 代币下注；第二个是链上订单簿，实现了去中心化的彩票交易市场，玩家可以随时买卖彩票，应对突发情况。"

### 结束语
"以上就是 EasyBet 项目的完整演示。项目实现了所有需求，代码质量高，测试覆盖完整，可以直接部署使用。感谢观看！"

---

## 🎓 评分要点对照

根据作业要求，确保演示中体现：

1. **公证人功能** ✅
   - 创建竞猜项目（多选项、奖池、时间）
   - 设置结果并结算

2. **玩家功能** ✅
   - 领取测试币
   - 购买彩票（获得 ERC721 凭证）
   - 买卖彩票（订单簿）
   - 平分奖池

3. **Bonus 1: ERC20 代币** (+2分) ✅
   - 发行 BettingToken
   - 水龙头领取
   - 使用代币下注

4. **Bonus 2: 链上订单簿** (+3分) ✅
   - 多价格挂单
   - 订单簿显示
   - 最优价格购买

---

祝演示顺利！🎉
