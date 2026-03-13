"""
准备FunctionGemma微调数据集
生成音乐控制相关的训练样本
"""

import json
import random
import os
from typing import List, Dict, Any
from pathlib import Path


class MusicDatasetGenerator:
    """生成音乐控制数据集"""
    
    def __init__(self):
        # 中文歌曲和艺术家数据（充分扩充版）
        self.songs = [
            # 周杰伦系列
            "稻香", "青花瓷", "七里香", "晴天", "简单爱", "夜曲", "听妈妈的话", 
            "彩虹", "星晴", "发如雪", "告白气球", "等你下课", "说好不哭", 
            "Mojito", "不能说的秘密", "以父之名", "东风破", "菊花台", "千里之外",
            "搁浅", "园游会", "甜甜的", "手写的从前", "爱在西元前", "止战之殇",
            
            # 林俊杰系列
            "江南", "曹操", "一千年以后", "美人鱼", "背对背拥抱", "她说", "小酒窝",
            "修炼爱情", "当你", "醉赤壁", "不为谁而作的歌", "可惜没如果", "杀手",
            
            # 陈奕迅系列
            "十年", "浮夸", "爱情转移", "好久不见", "不如这样", "淘汰", "K歌之王",
            "富士山下", "红玫瑰", "白玫瑰", "单车", "最佳损友",
            
            # 薛之谦系列
            "演员", "认真的雪", "丑八怪", "刚刚好", "违背的青春", "绅士", "暧昧",
            "你还要我怎样", "我好像在哪见过你", "天外来物",
            
            # 邓紫棋系列
            "光年之外", "泡沫", "来自天堂的魔鬼", "画", "倒数", "喜欢你", "后会无期",
            
            # 毛不易系列
            "消愁", "像我这样的人", "平凡的一天", "借", "无问", "二零三", "东北民谣",
            
            # 民谣系列
            "成都", "南山南", "董小姐", "理想", "斑马斑马", "模特", "春风十里",
            "安和桥", "傲寒", "南方姑娘", "画", "梵高先生", "温柔",
            
            # 经典老歌
            "后来", "勇气", "爱如潮水", "童话", "江南", "遇见", "我愿意",
            "挪威的森林", "传奇", "突然好想你", "小情歌", "你的背包", "星月神话",
            "月亮代表我的心", "甜蜜蜜", "但愿人长久", "千千阙歌", "吻别", "忘情水",
            
            # 近年流行
            "体面", "凉凉", "大鱼", "起风了", "下山", "芒种", "无羁", "陈情令",
            "红昭愿", "盗将行", "绿色", "世间美好与你环环相关", "可可托海的牧羊人",
            "踏山河", "万疆", "星辰大海", "孤勇者", "本草纲目", "错位时空",
            
            # 古风系列
            "赤伶", "琵琶行", "清明雨上", "半壶纱", "倾尽天下", "烟花易冷",
            "盗墓笔记·十年人间", "天行九歌", "权御天下", "牵丝戏", "离人愁",
            
            # 摇滚系列
            "海阔天空", "真的爱你", "光辉岁月", "不再犹豫", "喜欢你", "冷雨夜",
            "一生有你", "那些花儿", "平凡之路", "旅行", "夜空中最亮的星",
            
            # 说唱系列
            "野狼Disco", "热爱105度的你", "大碗宽面", "天干物燥", "不服",
            "中国话", "嘻哈庄脚情", "小酒窝", "Rap God",
            
            # 情歌系列
            "七月上", "可不可以", "关键词", "心如止水", "有点甜", "惊鸿一面",
            "山楂树之恋", "爱的初体验", "遗憾", "说散就散", "想见你想见你想见你",
            
            # 影视金曲
            "我的歌声里", "时间都去哪儿了", "父亲", "母亲", "朋友", "一路向北",
            "匆匆那年", "再见", "明天会更好", "朋友别哭", "祝福", "外婆的澎湖湾",
            
            # 快手热门
            "纸短情长", "我们不一样", "最美的期待", "38度6", "沙漠骆驼",
            "空空如也", "你的答案", "火红的萨日朗", "站着等你三千年",
            
            # 抖音热歌
            "少年", "惊雷", "世界这么大还是遇见你", "四季予你", "清空",
            "桥边姑娘", "白月光与朱砂痣", "春娇与志明", "飞鸟和蝉",
            
            # 欧美金曲
            "Shape of You", "Faded", "Something Just Like This", "Closer",
            "Despacito", "See You Again", "Hello", "Rolling in the Deep",
            "Someone Like You", "All of Me", "Stay", "The Nights",
            
            # 日韩热歌
            "Lemon", "打上花火", "春よ来い", "星が瞬くこんな夜に",
            "Dynamite", "Butter", "Permission to Dance", "How You Like That"
        ]
        
        self.artists = [
            # 华语男歌手
            "周杰伦", "林俊杰", "陈奕迅", "薛之谦", "李荣浩", "毛不易", 
            "张学友", "王力宏", "罗大佑", "李宗盛", "刘德华", "张信哲",
            "任贤齐", "庾澄庆", "光良", "品冠", "曹格", "王杰",
            "张宇", "齐秦", "伍佰", "张震岳", "卢广仲", "吴青峰",
            "苏打绿", "朴树", "许巍", "汪峰", "郑钧", "窦唯",
            "李健", "张杰", "胡彦斌", "陶喆", "潘玮柏", "萧敬腾",
            "林宥嘉", "方大同", "李荣浩", "华晨宇", "毛不易", "周深",
            "张信哲", "阿杜", "张韶涵", "郁可唯",
            
            # 华语女歌手
            "邓紫棋", "蔡依林", "孙燕姿", "梁静茹", "田馥甄", "张靓颖",
            "杨丞琳", "S.H.E", "容祖儿", "莫文蔚", "王菲", "那英",
            "张惠妹", "林忆莲", "王心凌", "萧亚轩", "许茹芸", "戴佩妮",
            "梁咏琪", "郑秀文", "陈慧琳", "李玟", "张韶涵", "郁可唯",
            "周笔畅", "李宇春", "张碧晨", "袁娅维", "谭维维", "吉克隽逸",
            "张靓颖", "邓紫棋", "杨宗纬", "林宥嘉",
            
            # 乐队/组合
            "五月天", "TFBOYS", "Beyond", "信乐团", "羽泉", "凤凰传奇",
            "筷子兄弟", "水木年华", "纵贯线", "旅行团", "海龟先生",
            
            # 民谣歌手
            "赵雷", "宋冬野", "马頔", "陈粒", "李志", "万晓利",
            "好妹妹", "程璧", "房东的猫", "花粥", "郭顶",
            
            # 说唱歌手
            "GAI", "VAVA", "小青龙", "欧阳靖", "热狗", "潘玮柏",
            
            # 港台经典
            "张国荣", "Beyond黄家驹", "谭咏麟", "梅艳芳", "邓丽君",
            "凤飞飞", "张学友", "刘德华", "郭富城", "黎明",
            
            # 欧美歌手
            "Ed Sheeran", "Taylor Swift", "Adele", "Justin Bieber",
            "Ariana Grande", "The Weeknd", "Bruno Mars", "Rihanna",
            
            # 日韩歌手
            "米津玄师", "YOASOBI", "BTS防弹少年团", "BLACKPINK", "IU",
            "TWICE", "EXO", "少女时代"
        ]
        
        # 命令模板
        self.play_templates = [
            "播放{song}",
            "我想听{song}",
            "放一首{song}",
            "来一首{song}",
            "播放{artist}的{song}",
            "我要听{artist}唱的{song}",
            "放{artist}的{song}",
            "打开{song}",
            "给我放{song}",
            "帮我播放{song}"
        ]
        
        self.pause_templates = [
            "暂停", "暂停播放", "暂停音乐", "停止播放", "先暂停一下"
        ]
        
        self.resume_templates = [
            "继续播放", "继续", "播放", "恢复播放", "开始播放"
        ]
        
        self.next_templates = [
            "下一首", "切歌", "换一首", "下一曲", "跳过这首"
        ]
        
        self.previous_templates = [
            "上一首", "返回上一首", "上一曲", "回到上一首"
        ]
        
        self.volume_templates = [
            "音量调到{level}",
            "把音量设置为{level}",
            "音量{level}",
            "声音调到{level}"
        ]
        
    def generate_function_call(self, function_name: str, parameters: Dict[str, Any]) -> str:
        """生成函数调用的JSON格式"""
        return json.dumps({
            "function": function_name,
            "parameters": parameters
        }, ensure_ascii=False)
    
    def create_training_sample(self, user_input: str, function_call: str) -> Dict[str, str]:
        """创建训练样本，使用FunctionGemma格式"""
        # FunctionGemma使用特殊的格式标记函数调用
        prompt = f"""<start_of_turn>user
{user_input}<end_of_turn>
<start_of_turn>model
<function_call>{function_call}</function_call><end_of_turn>"""
        
        return {
            "text": prompt,
            "user_input": user_input,
            "function_call": function_call
        }
    
    def generate_play_samples(self, num_samples: int) -> List[Dict]:
        """生成播放歌曲的样本"""
        samples = []
        for _ in range(num_samples):
            template = random.choice(self.play_templates)
            song = random.choice(self.songs)
            
            if "{artist}" in template:
                artist = random.choice(self.artists)
                user_input = template.format(song=song, artist=artist)
                function_call = self.generate_function_call(
                    "play_song",
                    {"song_name": song, "artist": artist}
                )
            else:
                user_input = template.format(song=song)
                function_call = self.generate_function_call(
                    "play_song",
                    {"song_name": song}
                )
            
            samples.append(self.create_training_sample(user_input, function_call))
        
        return samples
    
    def generate_control_samples(self) -> List[Dict]:
        """生成控制命令样本"""
        samples = []
        
        # 暂停
        for template in self.pause_templates:
            function_call = self.generate_function_call("pause_music", {})
            samples.append(self.create_training_sample(template, function_call))
        
        # 继续
        for template in self.resume_templates:
            function_call = self.generate_function_call("resume_music", {})
            samples.append(self.create_training_sample(template, function_call))
        
        # 下一首
        for template in self.next_templates:
            function_call = self.generate_function_call("next_song", {})
            samples.append(self.create_training_sample(template, function_call))
        
        # 上一首
        for template in self.previous_templates:
            function_call = self.generate_function_call("previous_song", {})
            samples.append(self.create_training_sample(template, function_call))
        
        # 音量控制
        for template in self.volume_templates:
            for _ in range(20):
                level = random.randint(0, 100)
                user_input = template.format(level=level)
                function_call = self.generate_function_call(
                    "set_volume",
                    {"level": level}
                )
                samples.append(self.create_training_sample(user_input, function_call))
        
        return samples
    
    def generate_dataset(self, num_play_samples: int = 5000) -> tuple:
        """生成完整数据集"""
        print("开始生成数据集...")
        
        # 生成所有样本
        all_samples = []
        all_samples.extend(self.generate_play_samples(num_play_samples))
        all_samples.extend(self.generate_control_samples())
        
        # 打乱数据
        random.shuffle(all_samples)
        
        # 分割训练集和验证集 (90/10)
        split_idx = int(len(all_samples) * 0.9)
        train_samples = all_samples[:split_idx]
        eval_samples = all_samples[split_idx:]
        
        print(f"生成完成: 训练样本 {len(train_samples)}, 验证样本 {len(eval_samples)}")
        
        return train_samples, eval_samples
    
    def save_dataset(self, samples: List[Dict], output_file: str):
        """保存数据集为JSONL格式"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"数据集已保存到: {output_file}")


def main():
    """主函数"""
    generator = MusicDatasetGenerator()
    
    # 生成数据集
    train_samples, eval_samples = generator.generate_dataset(num_play_samples=5000)
    
    # 保存数据集
    data_dir = Path(__file__).parent / "data"
    generator.save_dataset(train_samples, data_dir / "music_control_train.jsonl")
    generator.save_dataset(eval_samples, data_dir / "music_control_eval.jsonl")
    
    # 打印一些样本示例
    print("\n=== 训练样本示例 ===")
    for i, sample in enumerate(train_samples[:3], 1):
        print(f"\n样本 {i}:")
        print(f"输入: {sample['user_input']}")
        print(f"函数调用: {sample['function_call']}")


if __name__ == "__main__":
    main()
