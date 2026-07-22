"""生成演示数据 —— 知识库文档 + 用户消费数据"""
import json
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent

# ============ 知识库文档 ============

PRODUCT_MANUAL = """
# 数码产品知识手册

## 蓝牙耳机类
- 蓝牙版本: 5.3, 支持 SBC/AAC/LDAC 编码
- 续航时间: 单次充电续航 6-8 小时，充电盒可额外提供 24 小时续航
- 防水等级: IPX5（防汗防雨，不可浸泡）
- 降噪功能: 主动降噪 ANC，降噪深度 -35dB
- 充电接口: Type-C，支持快充（充电10分钟使用2小时）
- 兼容性: 支持 iOS/Android/Windows 多平台
- 保修政策: 1年质保，7天无理由退货，15天质量问题换货

## 智能手表类
- 屏幕: 1.43英寸 AMOLED，分辨率 466x466
- 电池: 300mAh，典型使用续航 14 天，重度使用 7 天
- 传感器: 心率、血氧、加速度计、陀螺仪、气压计
- 防水: 5ATM（50米防水，可游泳佩戴）
- 系统: 支持 iOS 12+ 和 Android 8+
- 功能: 消息通知、运动监测、睡眠分析、NFC支付
- 保修政策: 1年质保，7天无理由退货，15天质量问题换货

## 充电宝类
- 容量: 10000mAh/20000mAh 两种规格
- 输出: USB-A x2 (最高22.5W), USB-C x1 (最高65W双向快充)
- 输入: USB-C 65W，约2.5小时充满20000mAh
- 协议: 支持 PD3.0/QC4+/SCP/FCP/AFC 等多协议
- 安全: 过充保护、过放保护、温度保护、短路保护
- 重量: 10000mAh约200g，20000mAh约390g
- 保修政策: 18个月质保，7天无理由退货

## 手机壳类
- 材质: TPU硅胶 / PC硬壳 / 真皮 / 凯夫拉
- 兼容型号: iPhone 13/14/15系列，华为 Mate/P 系列，小米数字系列
- 特点: 防摔气囊设计，镜头保护凸起，精准孔位
- 清洁: 硅胶壳可用湿布擦拭，真皮壳避免接触酒精
- 保修政策: 质量问题30天换新，7天无理由退货

## 数据线/充电器类
- 接口: USB-C to USB-C, USB-C to Lightning, USB-A to Lightning
- 长度: 0.3m/1m/2m
- 功率: 支持最高100W PD快充
- 材质: 编织线身，铝合金接头
- 保修政策: 终身质保（非人为损坏），7天无理由退货
"""

RETURN_POLICY = """
# 电商平台退换货政策

## 通用退换货规则

### 无理由退货
- 签收后 **7天内** 支持无理由退货
- 需保证商品包装完整、配件齐全、商品无损坏
- 以下商品不支持无理由退货:
  - 已拆封的个人卫生用品（耳机、牙刷等）
  - 已激活的软件/游戏
  - 定制类商品
  - 鲜活易腐商品

### 质量问题退货
- 签收后 **15天内** 出现质量问题可申请换货
- 签收后 **30天内** 出现质量问题可申请维修
- 需提供质量问题证明（照片/视频）
- 检测确认为质量问题后，来回运费由商家承担

### 退款方式
- 原路退回: 支付宝/微信支付 3-5 个工作日
- 银行卡: 5-10 个工作日
- 平台余额: 即时到账

## 按品类细分

### 数码配件（耳机/手表/充电宝）
- 无理由退货: 签收后7天内（需包装完好，配件齐全）
- 质量问题: 签收后15天内换货，1年质保
- 特别注意: 蓝牙耳机因卫生原因，拆封后不支持无理由退货，请确认需求后购买

### 手机配件（手机壳/贴膜/数据线）
- 无理由退货: 签收后7天内（需未使用、包装完好）
- 质量问题: 30天内换新（手机壳），终身质保（数据线，非人为损坏）
- 贴膜类商品一经使用不支持退货

### 服装类
- 无理由退货: 签收后7天内（需吊牌完整、未洗涤、未穿着）
- 尺码问题: 支持换货（来回运费买家承担）
- 质量问题: 签收后15天内换货

### 美妆个护
- 无理由退货: 签收后7天内（需未拆封、包装完好）
- 过敏退货: 签收后30天内提供过敏证明可退货
- 已拆封的化妆品不支持无理由退货

## 特殊情况
- 大型促销活动期间（618、双11等）退货时效延长至15天
- PLUS会员享受专属退货通道和双倍退货时效
- 跨境商品退货需扣减关税和运费
"""

FAQ_DOC = """
# 电商客服常见问题 FAQ

## 物流相关

Q: 什么时候发货？
A: 现货商品下单后 24小时内发货，预售商品以页面标注时间为准。发货后会短信通知您快递单号。

Q: 可以指定快递公司吗？
A: 目前默认使用顺丰和京东快递，暂不支持指定其他快递。如需发其他快递请联系客服备注。

Q: 物流信息多久更新一次？
A: 快递揽收后通常 2-4小时更新一次物流信息。如超过24小时未更新，请联系客服查询。

Q: 可以修改收货地址吗？
A: 未发货的订单可以在订单详情中直接修改。已发货的订单需要联系客服拦截快递或转寄。

## 支付相关

Q: 支持哪些支付方式？
A: 支持支付宝、微信支付、银联卡、花呗分期、白条支付。

Q: 为什么支付失败？
A: 可能原因: 银行卡余额不足、超出单笔限额、网络异常。建议更换支付方式重试或联系银行客服。

Q: 如何申请发票？
A: 在订单详情页点击「申请发票」，可选择电子发票或纸质发票。电子发票即时生成，纸质发票随货发出。

## 售后相关

Q: 如何申请退货？
A: 在「我的订单」中找到对应订单，点击「申请退货」，填写退货原因并上传凭证，审核通过后按指引寄回商品即可。

Q: 退货后多久能收到退款？
A: 仓库签收退货商品后 1-2 个工作日完成质检，质检通过后支付宝/微信 3-5 个工作日到账。

Q: 换货流程是怎样的？
A: 申请换货 → 寄回商品 → 仓库签收质检 → 发新商品。整个流程通常需要 5-7 个工作日。

Q: 收到商品与描述不符怎么办？
A: 请先拍照保留证据，联系客服并提供订单号和照片。如确认是商家问题，退货来回运费由商家承担。

## 商品相关

Q: 如何查看商品详细规格？
A: 在商品详情页下滑查看「规格参数」板块，或联系客服获取产品手册。

Q: 蓝牙耳机可以连接多台设备吗？
A: 支持同时连接2台设备（手机+电脑），可一键切换音频来源。

Q: 充电宝可以带上飞机吗？
A: 额定能量不超过100Wh（约27000mAh）的充电宝可以随身携带，不可托运。10000mAh和20000mAh都可以携带。

## 会员相关

Q: PLUS会员有什么权益？
A: 包邮、专属折扣、双倍退货时效、优先客服、生日礼包等。

Q: 积分怎么获取和使用？
A: 购物1元=1积分，100积分=1元，可在下单时抵扣。积分有效期为1年。
"""

# ============ SQLite 数据库 ============

USERS = [
    {"user_id": "U10001", "name": "张明", "level": "PLUS", "phone": "138****1234", "register_date": "2024-03-15", "total_orders": 47, "total_spent": 12580.50},
    {"user_id": "U10002", "name": "李红", "level": "普通", "phone": "139****5678", "register_date": "2024-06-20", "total_orders": 12, "total_spent": 2340.00},
    {"user_id": "U10003", "name": "王芳", "level": "VIP", "phone": "136****9012", "register_date": "2023-11-08", "total_orders": 85, "total_spent": 28900.75},
    {"user_id": "U10004", "name": "赵伟", "level": "普通", "phone": "137****3456", "register_date": "2025-01-10", "total_orders": 3, "total_spent": 456.00},
    {"user_id": "U10005", "name": "陈静", "level": "PLUS", "phone": "135****7890", "register_date": "2024-08-25", "total_orders": 31, "total_spent": 8920.30},
]

PRODUCTS = {
    "P001": {"name": "蓝牙耳机 Pro", "category": "数码配件", "subcategory": "蓝牙耳机", "price": 299, "stock": 500},
    "P002": {"name": "智能手表 S3", "category": "数码配件", "subcategory": "智能手表", "price": 899, "stock": 200},
    "P003": {"name": "快充充电宝 20000mAh", "category": "数码配件", "subcategory": "充电宝", "price": 159, "stock": 800},
    "P004": {"name": "液态硅胶手机壳", "category": "手机配件", "subcategory": "手机壳", "price": 39, "stock": 2000},
    "P005": {"name": "编织数据线 2m", "category": "手机配件", "subcategory": "数据线", "price": 29, "stock": 3000},
    "P006": {"name": "真皮手机壳", "category": "手机配件", "subcategory": "手机壳", "price": 129, "stock": 300},
    "P007": {"name": "无线充电器 15W", "category": "数码配件", "subcategory": "充电器", "price": 79, "stock": 600},
    "P008": {"name": "运动T恤", "category": "服装", "subcategory": "上衣", "price": 99, "stock": 1500},
    "P009": {"name": "防晒霜 SPF50", "category": "美妆个护", "subcategory": "防晒", "price": 69, "stock": 1000},
    "P010": {"name": "充电宝 10000mAh", "category": "数码配件", "subcategory": "充电宝", "price": 99, "stock": 600},
}


def generate_orders():
    """为用户生成模拟订单数据"""
    orders = []
    order_id = 1000
    base_date = datetime(2024, 3, 1)

    for user in USERS:
        num_orders = random.randint(5, 20)
        user_products = random.choices(list(PRODUCTS.keys()), k=num_orders)
        for _ in range(num_orders):
            pid = random.choice(user_products)
            product = PRODUCTS[pid]
            days_ago = random.randint(1, 365)
            order_date = base_date + timedelta(days=days_ago)

            status = random.choice(["已完成", "已签收", "已发货", "退货中", "已完成", "已完成", "已完成"])
            amount = product["price"] * random.randint(1, 3)

            orders.append({
                "order_id": f"ORD{order_id:06d}",
                "user_id": user["user_id"],
                "product_id": pid,
                "product_name": product["name"],
                "category": product["category"],
                "amount": amount,
                "status": status,
                "order_date": order_date.strftime("%Y-%m-%d"),
                "payment_method": random.choice(["支付宝", "微信支付", "银行卡"]),
            })
            order_id += 1

    return orders


def init_database():
    """初始化 SQLite 数据库"""
    db_path = DATA_DIR / "db" / "shopmind.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT, level TEXT, phone TEXT,
            register_date TEXT, total_orders INTEGER, total_spent REAL
        );
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            user_id TEXT, product_id TEXT, product_name TEXT,
            category TEXT, amount REAL, status TEXT,
            order_date TEXT, payment_method TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id);
        CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
    """)

    # 插入用户
    for u in USERS:
        conn.execute(
            "INSERT OR REPLACE INTO users VALUES (?,?,?,?,?,?,?)",
            (u["user_id"], u["name"], u["level"], u["phone"], u["register_date"], u["total_orders"], u["total_spent"])
        )

    # 插入订单
    orders = generate_orders()
    for o in orders:
        conn.execute(
            "INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
            (o["order_id"], o["user_id"], o["product_id"], o["product_name"], o["category"], o["amount"], o["status"], o["order_date"], o["payment_method"])
        )

    conn.commit()
    conn.close()

    # 统计
    total_revenue = sum(o["amount"] for o in orders)
    print(f"数据库初始化完成:")
    print(f"  用户: {len(USERS)} 人")
    print(f"  商品: {len(PRODUCTS)} 种")
    print(f"  订单: {len(orders)} 笔")
    print(f"  总交易额: {total_revenue:.2f} 元")
    return orders


def save_knowledge_docs():
    """保存知识库文档到本地文件"""
    kb_dir = DATA_DIR / "knowledge"
    kb_dir.mkdir(parents=True, exist_ok=True)

    docs = {
        "product_manual.txt": PRODUCT_MANUAL,
        "return_policy.txt": RETURN_POLICY,
        "faq.txt": FAQ_DOC,
    }

    for filename, content in docs.items():
        path = kb_dir / filename
        path.write_text(content, encoding="utf-8")
        print(f"  知识库文档: {path} ({len(content)} 字)")

    return kb_dir


def generate_all():
    print("=" * 50)
    print("  ShopMind 演示数据生成")
    print("=" * 50)
    print()
    print("[1/2] 生成知识库文档...")
    kb_dir = save_knowledge_docs()
    print()
    print("[2/2] 初始化用户/订单数据...")
    orders = init_database()
    print()
    print("数据生成完成！")
    return kb_dir, orders


if __name__ == "__main__":
    generate_all()
