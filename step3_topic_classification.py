# -*- coding: utf-8 -*-
"""
Step 3: 15-Topic Taxonomy Classification
========================================
Uses DeepSeek LLM to classify health science articles into 15 categories:

 1. Cancer and Tumor Prevention & Treatment
 2. Chronic Disease and Metabolic Disease Management
 3. Cardiocerebrovascular Diseases and Emergency Warning
 4. Respiratory and ENT Health
 5. Infectious Disease Prevention and Control
 6. Maternal, Pediatric and Adolescent Health
 7. Organ Transplantation and Special Medical Procedures
 8. Traditional Chinese Medicine and Wellness
 9. Nutrition and Lifestyle
10. Musculoskeletal and Rehabilitation
11. Mental and Sleep Health
12. Environmental and Preventive Health
13. Reproductive and Sexual Health
14. Oral Health
15. Other

Configuration:
  - DEEPSEEK_API_KEY (env var)
  - EXCEL_PATH (data file path)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import sys
import io
import os
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("Step 3: 15-Topic Classification")
print("=" * 70)

# ==================== Topic Taxonomy ====================
TOPIC_TAXONOMY = {
    "Cancer and Tumor Prevention & Treatment": {
        "en": "癌症与肿瘤防治",
        "description": (
            "各类癌症（肺癌、肝癌、乳腺癌、胃癌、结直肠癌、前列腺癌、甲状腺癌、宫颈癌等）"
            "的早筛、诊断、治疗（手术/放疗/化疗/靶向/免疫）、康复及基因检测、肿瘤标志物"
        ),
        "keywords": [
            "癌症", "恶性肿瘤", "癌", "瘤", "早筛", "防癌", "致癌", "高危人群",
            "手术", "切除", "根治术", "分期", "恶性", "良性", "转移", "复发",
            "放疗", "化疗", "靶向", "免疫治疗", "PD-1", "PD-L1", "CAR-T", "内分泌治疗",
            "肿瘤标志物", "CEA", "CA125", "CA19-9", "AFP", "基因检测", "基因突变", "基因测序",
            "肺癌", "肺结节", "磨玻璃结节", "低剂量CT", "肺腺癌", "肺鳞癌", "小细胞肺癌",
            "胸腔镜", "肺叶切除", "胸外科",
            "胃癌", "胃镜", "幽门螺杆菌", "HP", "胃部不适",
            "结直肠癌", "肠癌", "肠镜", "息肉", "肠镜检查", "直肠癌",
            "肝癌", "甲胎蛋白", "肝血管瘤", "肝切除",
            "食管癌", "食管",
            "乳腺癌", "乳腺结节", "乳腺钼靶", "BI-RADS", "HER2", "ER", "PR", "乳腺外科",
            "甲状腺癌", "甲状腺结节", "甲功", "TSH", "甲外科",
            "宫颈癌", "TCT", "HPV", "阴道镜",
            "卵巢癌", "卵巢囊肿", "子宫肌瘤",
            "前列腺癌", "前列腺", "PSA", "穿刺", "泌尿外科",
            "肾癌", "膀胱癌", "淋巴瘤", "白血病", "骨髓移植",
            "神经内分泌肿瘤", "间质瘤",
            "便血", "呕血", "黄疸", "消瘦",
        ]
    },

    "Traditional Chinese Medicine and Wellness": {
        "en": "中医中药与养生",
        "description": (
            "中医理论、辨证论治、针灸、拔罐、推拿、正骨等传统中医疗法；"
            "中药材、方剂、药膳、膏方；以及中医养生保健（节气养生、体质调理等）"
        ),
        "keywords": [
            "中医", "中药", "中医理论", "辨证论治", "阴阳", "五行", "气血", "经络",
            "穴位", "脏腑", "肝郁", "脾虚", "肾虚", "湿热", "气虚", "血瘀",
            "体质", "体质辨识", "九种体质", "阳虚", "阴虚",
            "针灸", "针刺", "艾灸", "温针灸", "电针", "针刀", "浮针",
            "拔罐", "拔火罐", "走罐", "刺络拔罐",
            "推拿", "正骨", "整脊", "整复", "按摩", "小儿推拿",
            "刮痧", "耳穴", "埋线", "穴位注射",
            "中药", "中草药", "方剂", "经方", "时方", "药方",
            "人参", "黄芪", "枸杞", "当归", "党参", "三七", "灵芝", "天麻",
            "药膳", "食疗", "养生茶", "膏方", "代煎",
            "中药材", "中药饮片", "中药配方颗粒",
            "养生", "节气养生", "三伏贴", "三九灸", "冬病夏治",
            "体质调理", "亚健康", "治未病", "中医保健",
            "八段锦", "太极拳", "五禽戏",
        ]
    },

    "Nutrition and Lifestyle": {
        "en": "营养与生活方式",
        "description": (
            "科学饮食、营养知识、膳食指南、运动锻炼、体重管理、减重瘦身、"
            "戒烟限酒、健康体检及一般性预防保健内容"
        ),
        "keywords": [
            "饮食", "膳食", "营养", "营养素", "蛋白质", "膳食纤维", "碳水", "脂肪",
            "地中海饮食", "DASH饮食", "得舒饮食", "生酮", "轻断食", "168饮食",
            "低盐", "低脂", "低糖", "低热量", "少油", "少盐少糖",
            "蔬菜水果", "粗粮", "全谷物", "坚果", "Omega-3", "补铁", "补钙",
            "保健品", "膳食补充剂", "维生素", "复合维生素", "鱼油", "辅酶Q10",
            "运动", "锻炼", "健身", "有氧运动", "无氧运动", "抗阻训练",
            "跑步", "快走", "走路", "游泳", "骑行", "八段锦", "太极拳", "瑜伽",
            "力量训练", "核心训练", "柔韧性", "运动处方",
            "久坐", "久站", "上班族", "办公室",
            "减重", "减肥", "瘦身", "体重管理", "体脂率", "BMI", "内脏脂肪",
            "代餐", "轻食", "热量缺口", "燃脂", "减脂", "GLP-1",
            "限酒", "戒酒", "酒精", "饮酒", "酗酒", "醉酒",
            "体检", "健康体检", "年度体检", "筛查", "预防性检查",
            "健康生活方式", "亚健康", "治未病",
        ]
    },

    "Musculoskeletal and Rehabilitation": {
        "en": "骨骼肌肉与康复",
        "description": (
            "膝关节、肩周、腰椎、颈椎、骨质疏松、骨折、运动损伤的诊断治疗，"
            "以及康复训练、防跌倒指导、骨科术后康复"
        ),
        "keywords": [
            "膝关节", "膝盖", "半月板", "交叉韧带", "关节镜", "关节疼痛", "关节积液",
            "肩周炎", "肩袖", "肩关节", "冻结肩", "五十肩", "网球肘", "高尔夫球肘",
            "腰椎", "腰突", "腰椎间盘突出", "腰椎滑脱", "腰肌劳损", "腰痛", "腰疼",
            "颈椎病", "颈椎", "颈肩", "椎间盘突出", "脊髓型颈椎病", "神经根型",
            "骨折", "骨裂", "应力性骨折", "粉碎性骨折",
            "骨质疏松", "骨密度", "DXA", "补钙", "维生素D", "钙片", "骨量",
            "骨关节炎", "类风湿性关节炎", "RA", "强直性脊柱炎", "AS",
            "运动损伤", "扭伤", "拉伤", "肌肉拉伤", "韧带断裂", "跟腱断裂",
            "康复训练", "康复", "功能锻炼", "物理治疗", "PT", "作业治疗", "OT",
            "防跌倒", "跌倒", "平衡训练", "肌力训练", "步态训练",
            "骨科手术", "关节置换", "髋关节", "膝关节置换", "椎间盘置换",
            "椎间孔镜", "UBE", "微创", "术后康复",
        ]
    },

    "Chronic Disease and Metabolic Disease Management": {
        "en": "慢性病与代谢性疾病管理",
        "description": (
            "高血压、糖尿病、高血脂、高尿酸/痛风等慢性病的长期用药管理、指标监测、"
            "并发症预防；代谢综合征、脂肪肝等代谢性疾病的饮食运动指导与药物治疗"
        ),
        "keywords": [
            "慢病", "慢性病", "指标监测", "用药管理", "并发症", "合并症",
            "高血压", "血压", "降压", "收缩压", "舒张压", "高血压前期", "血压管理",
            "ARB", "ACEI", "CCB", "利尿剂", "心率",
            "糖尿病", "血糖", "胰岛素", "降糖", "糖化血红蛋白", "餐后血糖", "空腹血糖",
            "糖尿病前期", "糖耐量", "胰岛", "二甲双胍", "磺脲类", "糖友",
            "高血脂", "血脂", "胆固醇", "甘油三酯", "LDL", "HDL", "他汀",
            "代谢综合征",
            "高尿酸", "痛风", "嘌呤",
            "脂肪肝", "肝硬化",
        ]
    },

    "Cardiocerebrovascular Diseases and Emergency Warning": {
        "en": "心脑血管疾病与急症预警",
        "description": (
            "冠心病（心梗/心绞痛）、心律失常（房颤/早搏）、静脉血栓（深静脉血栓/肺栓塞）、"
            "脑卒中（脑梗/脑出血）、TIA 短暂性脑缺血发作等心脑血管急重症的早期识别、急救处理及康复二级预防；"
            "胸闷胸痛急性发作的预警信号与应急处置"
        ),
        "keywords": [
            "冠心病", "心梗", "心肌梗死", "心绞痛", "冠状动脉", "支架", "搭桥", "PCI",
            "心肌病", "心肌炎", "心包炎", "主动脉", "主动脉夹层",
            "心血管", "动脉粥样硬化", "动脉硬化", "心衰", "心力衰竭",
            "早搏", "房颤", "心律失常", "心电图", "ECG",
            "心脏超声", "冠脉CTA", "冠脉造影",
            "深静脉血栓", "肺栓塞", "静脉曲张",
            "脑卒中", "中风", "脑梗", "脑出血", "脑血管", "TIA",
            "短暂性脑缺血", "蛛网膜", "颅内动脉瘤",
            "胸闷", "胸痛", "心源性", "猝死",
            "急救", "120", "心肺复苏", "CPR",
        ]
    },

    "Respiratory and ENT Health": {
        "en": "呼吸系统与五官健康",
        "description": (
            "呼吸科：流感、新冠、感冒、肺炎、支气管炎、哮喘等呼吸道感染性疾病；"
            "五官科：鼻炎、鼻窦炎、咽炎、中耳炎、耳鸣等耳鼻喉疾病；"
            "眼科：视力、白内障、青光眼、黄斑病变；"
            "皮肤科：皮炎、湿疹、荨麻疹、痤疮等常见皮肤病"
        ),
        "keywords": [
            "流感", "甲流", "乙流", "禽流感", "感冒", "上呼吸道感染",
            "新冠", "COVID", "阳性", "核酸", "抗原", "隔离", "Omicron",
            "肺炎", "细菌性肺炎", "病毒性肺炎", "支原体肺炎", "白肺",
            "支气管炎", "毛细支气管炎", "支气管哮喘", "哮喘",
            "咳嗽", "咽痛", "咽拭子", "发热", "退烧",
            "抗病毒", "抗生素", "头孢", "阿莫西林", "阿奇霉素",
            "眼科", "视力", "视力下降", "视力模糊", "配镜", "验光",
            "白内障", "青光眼", "黄斑病变", "黄斑变性", "眼底病变",
            "干眼症", "干眼", "眼疲劳", "结膜炎", "红眼病",
            "飞蚊症", "视网膜脱离", "玻璃体混浊", "白内障手术",
            "耳鼻喉", "五官",
            "鼻炎", "过敏性鼻炎", "萎缩性鼻炎", "鼻窦炎", "鼻息肉", "鼻中隔偏曲",
            "咽炎", "慢性咽炎", "咽喉炎", "扁桃体", "扁桃体炎",
            "中耳炎", "分泌性中耳炎", "化脓性中耳炎", "耳鸣", "耳聋", "听力下降",
            "眩晕", "美尼尔", "前庭", "耳石症", "位置性眩晕",
            "腺样体", "腺样体肥大", "打鼾", "睡眠呼吸暂停",
            "喉癌", "声带", "声带息肉", "喉镜",
            "皮肤", "皮肤科", "皮炎", "湿疹", "荨麻疹", "痤疮", "青春痘",
            "银屑病", "白癜风", "脂溢性皮炎",
            "皮肤过敏", "化妆品过敏", "接触性皮炎", "虫咬皮炎",
            "疣", "寻常疣", "跖疣",
            "激光美容", "光子嫩肤", "果酸换肤",
            "瘙痒", "皮肤干燥", "皮肤感染",
        ]
    },

    "Infectious Disease Prevention and Control": {
        "en": "传染病与感染性疾病防控",
        "description": (
            "乙肝、丙肝、结核、艾滋病等慢性传染病的抗病毒治疗与管理；"
            "诺如病毒、手足口、狂犬病、带状疱疹、水痘等常见传染病的预防、症状及治疗；"
            "各类疫苗（流感/肺炎/乙肝/HPV/带状疱疹疫苗）的接种知识与科普"
        ),
        "keywords": [
            "乙肝", "丙肝", "病毒性肝炎", "肝炎", "抗病毒治疗",
            "结核", "肺结核", "痰检", "耐药结核",
            "艾滋病", "HIV", "梅毒", "淋病", "性传播",
            "诺如", "诺如病毒", "胃肠炎", "呕吐", "腹泻", "急性胃肠炎",
            "手足口", "疱疹性咽峡炎", "轮状病毒",
            "水痘", "腮腺炎", "风疹", "麻疹",
            "狂犬病", "狂犬疫苗", "破伤风",
            "带状疱疹", "蛇串疮",
            "传染", "感染", "接触史", "聚集性",
            "疫苗接种", "接种疫苗", "流感疫苗", "肺炎疫苗",
            "乙肝疫苗", "HPV疫苗", "宫颈癌疫苗", "带状疱疹疫苗",
            "脊灰疫苗", "百白破", "麻腮风", "水痘疫苗",
            "一类疫苗", "二类疫苗", "自费疫苗",
            "血常规", "CRP", "PCT", "降钙素原", "C反应蛋白",
        ]
    },

    "Maternal, Pediatric and Adolescent Health": {
        "en": "妇幼与儿童青少年健康",
        "description": (
            "产科：科学备孕、孕期管理（营养/产检/并发症）、分娩方式选择、产后康复及盆底肌修复；"
            "儿科：儿童常见病（发烧/咳嗽/腹泻）、疫苗接种、生长发育监测；"
            "乳腺健康（哺乳期）：哺乳期乳腺炎、母乳喂养指导、断奶护理；"
            "妇科疾病（子宫肌瘤、卵巢囊肿等非生殖健康类）"
        ),
        "keywords": [
            "备孕", "怀孕", "孕期", "孕早期", "孕中期", "孕晚期", "孕妇",
            "产检", "孕检", "唐筛", "无创DNA", "NT", "大排畸", "小排畸",
            "分娩", "顺产", "剖宫产", "无痛分娩", "导乐", "临产", "宫缩",
            "产后", "坐月子", "产后康复", "盆底肌", "腹直肌", "产褥期",
            "妊娠", "妊娠期糖尿病", "妊娠期高血压", "胎盘", "胎位",
            "妇科", "子宫肌瘤", "卵巢囊肿",
            "哺乳期", "乳腺炎", "乳腺增生", "乳腺结节", "乳腺健康",
            "母乳", "开奶", "断奶", "回奶", "追奶",
            "乳头", "乳汁", "堵奶", "奶结",
            "儿科", "儿童", "宝宝", "婴幼儿", "新生儿", "小儿",
            "发烧", "退热", "物理降温", "高热惊厥",
            "咳嗽", "腹泻", "便秘", "肠绞痛", "乳糖不耐受", "牛奶蛋白过敏",
            "生长发育", "身高", "体重", "智力发育", "早教",
            "预防接种", "一类疫苗", "二类疫苗", "自费疫苗",
            "宝妈", "宝爸", "准妈妈", "准爸爸",
        ]
    },

    "Organ Transplantation and Special Medical Procedures": {
        "en": "器官移植与特殊诊疗技术",
        "description": (
            "肾移植、肝移植、心脏移植、肺移植等器官移植手术及术后管理；"
            "血液透析（血透）、腹膜透析（腹透）等肾脏替代治疗；"
            "干细胞移植、细胞治疗等前沿特殊诊疗技术；"
            "以及 ICU 重症监护相关知识"
        ),
        "keywords": [
            "肾移植", "肝移植", "心脏移植", "肺移植", "器官移植",
            "移植手术", "移植后", "移植患者", "排斥反应", "免疫抑制",
            "活体移植", "遗体捐献", "器官捐献",
            "血液透析", "血透", "腹膜透析", "腹透", "透析",
            "透析患者", "透析通路", "动静脉内瘘", "透析导管",
            "肾功能衰竭", "尿毒症", "肾衰竭",
            "干细胞移植", "干细胞治疗", "细胞治疗", "CAR-T",
            "造血干细胞", "脐带血",
            "ICU", "重症监护", "生命体征", "危重症",
        ]
    },

    "Mental and Sleep Health": {
        "en": "心理与睡眠健康",
        "description": (
            "焦虑症、抑郁症、双相等情绪障碍的心理疏导与药物治疗；"
            "失眠、睡眠呼吸暂停等睡眠问题的改善方法；"
            "阿尔茨海默病（老年痴呆）、帕金森、面瘫、癫痫等神经精神系统疾病"
        ),
        "keywords": [
            "焦虑", "焦虑症", "焦虑障碍", "焦虑状态", "社交焦虑",
            "抑郁", "抑郁症", "抑郁状态", "情绪低落", "重度抑郁", "轻中度抑郁",
            "心理", "心理健康", "精神心理", "情绪", "心理疏导", "心理咨询",
            "双相情感", "躁郁症", "精神分裂", "PTSD",
            "失眠", "睡眠障碍", "睡不着", "早醒", "睡眠质量", "褪黑素", "安眠药",
            "打鼾", "睡眠呼吸暂停", "OSA", "阻塞性睡眠呼吸暂停",
            "阿尔茨海默", "老年痴呆", "认知障碍", "认知功能下降", "记忆力下降",
            "帕金森", "帕金森病", "静止性震颤", "肌强直", "运动迟缓",
            "面瘫", "面神经炎", "Bell麻痹",
            "癫痫", "抽搐", "惊厥",
            "神经痛", "三叉神经痛", "坐骨神经痛",
            "神经内科", "精神科", "心理科",
        ]
    },

    "Environmental and Preventive Health": {
        "en": "环境与预防健康",
        "description": (
            "环境污染（雾霾/PM2.5/甲醛/辐射）对健康的影响与防护；"
            "中毒急救（农药/食物/药物/一氧化碳中毒）的现场处置；"
            "职业健康（尘肺/噪声性耳聋/职业病）及消毒隔离防护指导"
        ),
        "keywords": [
            "雾霾", "PM2.5", "空气污染", "室内污染", "甲醛", "苯",
            "辐射", "X线辐射", "CT辐射", "手机辐射", "电磁辐射",
            "水污染", "重金属污染",
            "中毒", "农药中毒", "食物中毒", "药物中毒", "酒精中毒",
            "一氧化碳中毒", "煤气中毒", "亚硝酸盐中毒",
            "急救", "120", "心肺复苏", "CPR",
            "洗胃", "导泻", "解毒", "催吐",
            "烧伤", "烫伤", "电击伤", "溺水",
            "外伤", "割伤", "擦伤", "撕裂伤",
            "职业病", "尘肺", "噪声性耳聋", "职业危害",
            "高温作业", "粉尘", "苯中毒",
            "消毒", "灭菌", "隔离", "防护",
            "健康教育", "疾病预防",
        ]
    },

    "Reproductive and Sexual Health": {
        "en": "生殖与性健康",
        "description": (
            "性生活安全、避孕节育、不孕不育、辅助生殖技术（试管婴儿/冻卵）"
            "及性传播疾病（HPV/尖锐湿疣/梅毒/淋病）防治；"
            "多囊卵巢、卵巢早衰、月经不调、围绝经期等妇科内分泌疾病；"
            "男性少精弱精、前列腺炎等生殖健康问题"
        ),
        "keywords": [
            "避孕", "节育", "紧急避孕", "短效避孕", "皮下埋植",
            "上环", "取环", "安全期", "体外射精", "事后避孕",
            "性传播", "STD", "性疾病", "性疾病门诊",
            "HPV", "尖锐湿疣", "梅毒", "淋病", "衣原体", "支原体感染",
            "阴道炎", "盆腔炎", "附件炎", "盆腔积液",
            "不孕", "不育", "人工授精", "人工受孕",
            "试管婴儿", "体外受精", "IVF", "一代试管", "二代试管", "三代试管",
            "促排卵", "促排", "取卵", "移植", "鲜胚", "冻胚", "囊胚",
            "冻卵", "冻胚", "生育力保存",
            "PGD", "PGS", "胚胎植入", "着床", "胚胎移植",
            "供卵", "供精", "代孕",
            "多囊卵巢", "PCOS", "多囊", "月经不调", "月经紊乱",
            "闭经", "月经稀发", "功能失调性子宫出血", "功血",
            "围绝经期", "围绝经", "绝经", "更年期", "卵巢早衰", "POI", "POF",
            "卵巢储备", "卵巢功能",
            "卵巢", "卵子", "卵泡", "卵母细胞", "AMH", "基础卵泡",
            "输卵管", "子宫", "子宫内膜", "宫腔", "宫颈",
            "宫腔镜", "输卵管造影",
            "子宫内膜息肉", "宫腔粘连", "子宫内膜异位", "巧克力囊肿", "腺肌症",
            "子宫肌瘤剔除",
            "精子", "精液", "少精", "弱精", "无精", "畸精",
            "前列腺", "前列腺炎", "精索静脉曲张", "附睾",
            "胎停", "胚胎停育", "复发性流产", "习惯性流产",
        ]
    },

    "Oral Health": {
        "en": "口腔健康",
        "description": (
            "龋齿、蛀牙、补牙等牙体疾病；拔牙、智齿、种植牙等口腔外科；"
            "根管治疗、牙髓炎等牙髓疾病；牙齿矫正（正畸）、隐形矫正等美齿技术；"
            "牙周炎、牙龈炎、牙结石、洗牙等牙周健康"
        ),
        "keywords": [
            "口腔", "牙科", "牙齿", "龋齿", "蛀牙", "补牙",
            "拔牙", "智齿", "拔智齿", "多生牙",
            "口腔外科", "牙槽外科",
            "根管治疗", "牙神经", "牙髓炎", "牙髓坏死",
            "种植牙", "种牙", "烤瓷牙", "全瓷牙", "牙冠", "牙桥", "瓷贴面",
            "正畸", "牙齿矫正", "牙套", "隐形矫正", "舌侧矫正", "矫正器",
            "牙周炎", "牙龈炎", "牙周治疗", "牙结石", "洗牙", "洁牙",
            "牙龈出血", "牙龈萎缩", "牙周袋",
            "口腔溃疡", "口腔扁平苔藓", "口腔癌", "舌癌", "口腔黏膜",
            "牙科手术", "牙科检查", "牙片", "全景片", "CBCT",
            "牙科麻醉", "笑气镇静",
        ]
    },

    "Other": {
        "en": "其他",
        "description": "不属于上述十四类医疗健康内容，比例应控制在 3% 以内",
        "keywords": [
            "罕见病", "遗传病", "基因病",
        ]
    },
}

# ==================== API Configuration ====================
print("\n" + "=" * 70)
print("Step 3: 15-Topic Classification")
print("=" * 70)

DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    ""  # Set via environment variable: export DEEPSEEK_API_KEY="your-key"
)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash")

if DEEPSEEK_API_KEY:
    print(f"DEEPSEEK_API_KEY detected: {DEEPSEEK_API_KEY[:10]}...")
else:
    print("WARNING: DEEPSEEK_API_KEY not set. Classification will fail.")

from openai import OpenAI
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.siliconflow.cn/v1"
)
print(f"Using model: {DEEPSEEK_MODEL}")


def build_topic_prompt(text):
    """Build prompt for LLM classification."""
    topic_list = "\n".join([
        f"- {name} ({info['en']}): {info['description']}"
        for name, info in TOPIC_TAXONOMY.items()
    ])
    return f"""你是一个医疗科普内容分类专家。请根据以下文本内容，判断它属于哪个健康议题类别。

文本内容：
{text}

可选类别：
{topic_list}

注意：
1. 优先判断文本的主要话题是什么疾病/健康领域
2. 如果文本同时涉及多个类别，以最主要的主题为准，不要重复分类
3. 中医中药内容（针灸/拔罐/中药/药膳）归入 "Traditional Chinese Medicine and Wellness"
4. 疫苗接种（流感疫苗/HPV疫苗等）归入 "Infectious Disease Prevention and Control"
5. 胸闷/胸痛/急救归入 "Cardiocerebrovascular Diseases and Emergency Warning"
6. 过敏性鼻炎/哮喘/湿疹等临床过敏归入 "Respiratory and ENT Health"；环境污染/中毒/职业健康归入 "Environmental and Preventive Health"
7. 器官移植与透析相关内容归入 "Organ Transplantation and Special Medical Procedures"
8. 生殖健康（备孕/试管/多囊/月经/性传播疾病）归入 "Reproductive and Sexual Health"
9. 口腔健康（龋齿/种植牙/正畸/牙周炎）归入 "Oral Health"
10. 输出格式：只需输出一个类别名称，不要解释

直接输出类别名称："""


def classify_by_api(text, client):
    """Classify using DeepSeek API."""
    prompt = build_topic_prompt(text[:1500])  # First 1500 chars

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是一个医疗科普内容分类专家，只输出类别名称。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=60
        )
        result = response.choices[0].message.content.strip()

        if result in TOPIC_TAXONOMY:
            return result

        for name in TOPIC_TAXONOMY:
            if name in result or result in name:
                return name

        alias_map = {
            "肿瘤": "Cancer and Tumor Prevention & Treatment",
            "肺癌": "Cancer and Tumor Prevention & Treatment",
            "肝癌": "Cancer and Tumor Prevention & Treatment",
            "乳腺癌": "Cancer and Tumor Prevention & Treatment",
            "胃癌": "Cancer and Tumor Prevention & Treatment",
            "肠癌": "Cancer and Tumor Prevention & Treatment",
            "甲状腺": "Cancer and Tumor Prevention & Treatment",
            "宫颈癌": "Cancer and Tumor Prevention & Treatment",
            "前列腺癌": "Cancer and Tumor Prevention & Treatment",
            "放疗": "Cancer and Tumor Prevention & Treatment",
            "化疗": "Cancer and Tumor Prevention & Treatment",
            "靶向": "Cancer and Tumor Prevention & Treatment",
            "免疫治疗": "Cancer and Tumor Prevention & Treatment",
            "中医": "Traditional Chinese Medicine and Wellness",
            "中药": "Traditional Chinese Medicine and Wellness",
            "针灸": "Traditional Chinese Medicine and Wellness",
            "艾灸": "Traditional Chinese Medicine and Wellness",
            "拔罐": "Traditional Chinese Medicine and Wellness",
            "推拿": "Traditional Chinese Medicine and Wellness",
            "药膳": "Traditional Chinese Medicine and Wellness",
            "营养": "Nutrition and Lifestyle",
            "饮食": "Nutrition and Lifestyle",
            "运动": "Nutrition and Lifestyle",
            "减重": "Nutrition and Lifestyle",
            "减肥": "Nutrition and Lifestyle",
            "骨科": "Musculoskeletal and Rehabilitation",
            "康复": "Musculoskeletal and Rehabilitation",
            "骨骼": "Musculoskeletal and Rehabilitation",
            "呼吸": "Respiratory and ENT Health",
            "鼻炎": "Respiratory and ENT Health",
            "肺炎": "Respiratory and ENT Health",
            "流感": "Respiratory and ENT Health",
            "新冠": "Respiratory and ENT Health",
            "皮肤": "Respiratory and ENT Health",
            "高血压": "Chronic Disease and Metabolic Disease Management",
            "糖尿病": "Chronic Disease and Metabolic Disease Management",
            "高血脂": "Chronic Disease and Metabolic Disease Management",
            "痛风": "Chronic Disease and Metabolic Disease Management",
            "慢病": "Chronic Disease and Metabolic Disease Management",
            "冠心病": "Cardiocerebrovascular Diseases and Emergency Warning",
            "心梗": "Cardiocerebrovascular Diseases and Emergency Warning",
            "心绞痛": "Cardiocerebrovascular Diseases and Emergency Warning",
            "心衰": "Cardiocerebrovascular Diseases and Emergency Warning",
            "房颤": "Cardiocerebrovascular Diseases and Emergency Warning",
            "脑卒中": "Cardiocerebrovascular Diseases and Emergency Warning",
            "中风": "Cardiocerebrovascular Diseases and Emergency Warning",
            "脑梗": "Cardiocerebrovascular Diseases and Emergency Warning",
            "脑出血": "Cardiocerebrovascular Diseases and Emergency Warning",
            "肺栓塞": "Cardiocerebrovascular Diseases and Emergency Warning",
            "深静脉血栓": "Cardiocerebrovascular Diseases and Emergency Warning",
            "猝死": "Cardiocerebrovascular Diseases and Emergency Warning",
            "胸闷": "Cardiocerebrovascular Diseases and Emergency Warning",
            "胸痛": "Cardiocerebrovascular Diseases and Emergency Warning",
            "急救": "Cardiocerebrovascular Diseases and Emergency Warning",
            "120": "Cardiocerebrovascular Diseases and Emergency Warning",
            "心肺复苏": "Cardiocerebrovascular Diseases and Emergency Warning",
            "流感": "Infectious Disease Prevention and Control",
            "新冠": "Infectious Disease Prevention and Control",
            "肺炎": "Infectious Disease Prevention and Control",
            "乙肝": "Infectious Disease Prevention and Control",
            "丙肝": "Infectious Disease Prevention and Control",
            "结核": "Infectious Disease Prevention and Control",
            "艾滋病": "Infectious Disease Prevention and Control",
            "HIV": "Infectious Disease Prevention and Control",
            "梅毒": "Infectious Disease Prevention and Control",
            "带状疱疹": "Infectious Disease Prevention and Control",
            "疫苗接种": "Infectious Disease Prevention and Control",
            "流感疫苗": "Infectious Disease Prevention and Control",
            "肺炎疫苗": "Infectious Disease Prevention and Control",
            "传染": "Infectious Disease Prevention and Control",
            "怀孕": "Maternal, Pediatric and Adolescent Health",
            "孕期": "Maternal, Pediatric and Adolescent Health",
            "分娩": "Maternal, Pediatric and Adolescent Health",
            "儿科": "Maternal, Pediatric and Adolescent Health",
            "儿童": "Maternal, Pediatric and Adolescent Health",
            "妇科": "Maternal, Pediatric and Adolescent Health",
            "乳腺炎": "Maternal, Pediatric and Adolescent Health",
            "母乳": "Maternal, Pediatric and Adolescent Health",
            "哺乳期": "Maternal, Pediatric and Adolescent Health",
            "肾移植": "Organ Transplantation and Special Medical Procedures",
            "肝移植": "Organ Transplantation and Special Medical Procedures",
            "血液透析": "Organ Transplantation and Special Medical Procedures",
            "血透": "Organ Transplantation and Special Medical Procedures",
            "腹膜透析": "Organ Transplantation and Special Medical Procedures",
            "透析": "Organ Transplantation and Special Medical Procedures",
            "干细胞移植": "Organ Transplantation and Special Medical Procedures",
            "ICU": "Organ Transplantation and Special Medical Procedures",
            "焦虑": "Mental and Sleep Health",
            "抑郁": "Mental and Sleep Health",
            "心理": "Mental and Sleep Health",
            "失眠": "Mental and Sleep Health",
            "阿尔茨海默": "Mental and Sleep Health",
            "帕金森": "Mental and Sleep Health",
            "癫痫": "Mental and Sleep Health",
            "试管婴儿": "Reproductive and Sexual Health",
            "试管": "Reproductive and Sexual Health",
            "促排卵": "Reproductive and Sexual Health",
            "冻卵": "Reproductive and Sexual Health",
            "多囊": "Reproductive and Sexual Health",
            "卵巢早衰": "Reproductive and Sexual Health",
            "月经不调": "Reproductive and Sexual Health",
            "人工授精": "Reproductive and Sexual Health",
            "宫外孕": "Reproductive and Sexual Health",
            "胎停": "Reproductive and Sexual Health",
            "复发性流产": "Reproductive and Sexual Health",
            "性传播": "Reproductive and Sexual Health",
            "尖锐湿疣": "Reproductive and Sexual Health",
            "HPV": "Reproductive and Sexual Health",
            "避孕": "Reproductive and Sexual Health",
            "口腔": "Oral Health",
            "牙科": "Oral Health",
            "牙齿": "Oral Health",
            "龋齿": "Oral Health",
            "蛀牙": "Oral Health",
            "补牙": "Oral Health",
            "拔牙": "Oral Health",
            "智齿": "Oral Health",
            "种植牙": "Oral Health",
            "正畸": "Oral Health",
            "牙周炎": "Oral Health",
            "洗牙": "Oral Health",
            "根管治疗": "Oral Health",
            "过敏": "Environmental and Preventive Health",
            "中毒": "Environmental and Preventive Health",
            "急救": "Environmental and Preventive Health",
            "雾霾": "Environmental and Preventive Health",
            "甲醛": "Environmental and Preventive Health",
        }
        for alias, canonical in alias_map.items():
            if alias in result:
                return canonical

        return "Other"
    except Exception as e:
        print(f"  API error: {e}")
        return "Other"


# ==================== Load Data ====================
print("\nLoading data...")
EXCEL_PATH = r'.xlsx'
df = pd.read_excel(EXCEL_PATH)
print(f"Data: {len(df)} articles")

if 'Topic' not in df.columns:
    df['Topic'] = None
    print("Created 'Topic' column")
else:
    print("'Topic' column exists")

has_topic = df['Topic'].notna() & (df['Topic'] != '')
done_count = has_topic.sum()
need_mask = ~has_topic
total = need_mask.sum()
print(f"Already classified: {done_count}")
print(f"Need to process: {total}")

if total == 0:
    print("\nAll articles classified. Saving...")
    df.to_excel(EXCEL_PATH, index=False)
    print(f"Saved to: {EXCEL_PATH}")
    sys.exit(0)

# ==================== Run Classification ====================
print(f"\n{'='*70}")
print(f"Starting classification: {total} articles")
print(f"{'='*70}\n")

MAX_WORKERS = 10

topic_results = []
df_to_process = df[need_mask].copy()
progress_interval = 100
save_interval = 1000

def process_one(orig_idx, row):
    text = str(row.get('text', ''))
    topic = classify_by_api(text, client)
    return orig_idx, topic

processed = 0
all_tasks = [(orig_idx, row) for orig_idx, row in df_to_process.iterrows()]

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_one, orig_idx, row): orig_idx
               for orig_idx, row in all_tasks}

    for future in as_completed(futures):
        orig_idx, topic = future.result()
        df.at[orig_idx, 'Topic'] = topic
        topic_results.append(topic)

        processed += 1
        if processed % progress_interval == 0 or processed == total:
            pct = processed / total * 100
            recent = topic_results[-1]
            print(f"  [{processed:6d}/{total}] ({pct:5.1f}%) {recent}")

        if processed % save_interval == 0:
            df.to_excel(EXCEL_PATH, index=False)
            print(f"  [Auto-saved] {processed} articles")

# ==================== Statistics ====================
print(f"\n{'='*70}")
print("Classification Results")
print(f"{'='*70}")

topic_counts = pd.Series(topic_results).value_counts().sort_values(ascending=False)
total_topic = len(topic_results)

for topic, count in topic_counts.items():
    pct = count / total_topic * 100
    marker = " <<<" if pct >= 5 else ""
    print(f"  {topic:50s}: {count:6d} ({pct:5.1f}%){marker}")

other_pct = topic_counts.get("Other", 0) / total_topic * 100
print(f"\n  Other: {other_pct:.1f}%  (target <5%)")

# ==================== Save ====================
print(f"\n{'='*70}")
print("Saving")
print(f"{'='*70}")

df.to_excel(EXCEL_PATH, index=False)
print(f"Saved to: {EXCEL_PATH}")
print(f"  Topic column updated: {total} articles")

print(f"\n{'='*70}")
print("Step 3 Complete!")
print(f"{'='*70}")
