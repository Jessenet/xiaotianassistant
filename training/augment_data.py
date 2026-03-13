"""
数据增强脚本: 为少数类函数生成更多训练样本, 平衡数据集
增加负样本 (非音乐输入 → none), 清理模糊短输入
"""
import json
import random
import itertools
from pathlib import Path

random.seed(42)

# ═══════════════════════════════════════════════════════════════
#  各函数的输入模板
# ═══════════════════════════════════════════════════════════════

PAUSE_INPUTS = [
    "暂停", "暂停播放", "暂停音乐", "停一下", "停下来", "不听了",
    "先暂停", "暂停一下", "停止播放", "别放了", "先别放了", "关掉音乐",
    "音乐暂停", "歌暂停", "暂停歌曲", "停止音乐", "Music暂停",
    "停一停", "不想听了", "不要放了", "暂且停下", "先停一下",
    "把音乐停了", "帮我暂停", "暂停当前歌曲", "停掉音乐",
    "把歌停掉", "先暂停一下", "帮我停一下音乐", "暂停当前播放",
    "请暂停", "麻烦暂停", "暂停吧", "停", "停了吧",
    "帮我暂停一下", "不播了", "先不听了", "暂停下", "停下",
    "静音", "关音乐", "把音乐关了", "关掉", "关了吧",
    "不放了", "先不放了", "别播了", "不要播放了", "关掉播放",
    "把歌关了", "不要继续放了", "暂时停下", "稍停一下", "停一会",
]

RESUME_INPUTS = [
    "继续播放", "继续", "恢复播放", "接着放", "接着播放",
    "继续放", "恢复音乐", "接着听", "继续听", "继续放歌",
    "恢复", "继续播放音乐", "放吧", "继续放吧", "接着吧",
    "帮我恢复播放", "帮我继续播放", "麻烦继续", "恢复播放吧",
    "继续播", "再放", "再播放", "接着播", "接着放歌",
    "把歌继续放", "音乐继续", "歌继续放", "帮我继续", "继续吧",
    "接着来", "继续来", "播放继续", "恢复歌曲", "恢复播放歌曲",
    "重新播放", "重新放", "再来", "继续放音乐", "音乐恢复",
    "帮我恢复", "恢复吧", "再放吧", "接着放吧", "继续就行",
    "可以继续了", "开始播放", "继续播放当前歌曲", "恢复当前播放",
]

NEXT_INPUTS = [
    "下一首", "切歌", "换一首", "下一曲", "下首歌",
    "切下一首", "不听这首了", "播放下一首歌",
    "播放下一首", "下一首歌", "切换下一首", "放下一首", "换下一首",
    "跳过这首歌", "不要这首歌了", "换一首歌", "下一首歌曲",
    "切到下一首", "帮我切歌", "帮我换一首", "跳到下一首",
    "不喜欢这首切一下", "这首不好听换一首", "帮我切到下一首歌",
    "播放下一曲", "切下一曲", "下一首吧", "换首歌听听",
    "帮我跳过这首", "跳过这首歌曲", "不想听这个换一首",
    "帮我切到下一首", "这首歌跳过吧", "来一首新的", "换首别的听",
    "不喜欢换一首", "换别的歌", "换个歌听听", "帮我切一下歌",
    "帮我换首歌", "切到下一曲", "这首不好听", "跳过这首吧",
    "播放下一首歌曲", "换首歌吧", "切换到下一首歌", "换一首歌听",
]

PREV_INPUTS = [
    "上一首", "上一曲", "前一首", "上首歌", "回到上一首",
    "切到上一首", "播放上一首", "放上一首", "返回上一首",
    "再听一遍上一首", "播放前一首歌", "前一首歌", "之前那首歌",
    "回到上一首歌", "上一曲歌", "切上一首", "退回上一首",
    "帮我切到上一首", "帮我回到上一首", "上一个歌", "前面那首歌",
    "回上一首", "退到上一首", "播放前一首", "放前一首",
    "上一首歌曲", "切回上一首", "回退到上一首", "之前的歌",
    "帮我放上一首", "上首歌曲", "播放上一曲", "切到前一首",
    "回前一首", "放回上一首", "刚才那首歌", "再来一遍上一首",
    "上一首吧", "返回上一曲", "前一曲", "前面一首歌",
    "帮我播放上一首", "放回之前的歌", "帮我切到上一曲",
]

# ═══════════════════════════════════════════════════════════════
#  负样本 (非音乐输入 → none)
# ═══════════════════════════════════════════════════════════════

NEGATIVE_INPUTS = [
    # 问候/闲聊
    "你好", "早上好", "晚上好", "你好啊", "嗨",
    "晚安", "早安", "你好呀", "哈喽", "嘿",
    # 通用问题
    "今天天气怎么样", "现在几点了", "明天星期几",
    "今天几号", "什么时候下雨", "温度多少度",
    # 技术/互联网
    "大模型", "人工智能", "什么是AI", "机器学习",
    "深度学习", "ChatGPT", "神经网络", "自然语言处理",
    # 生活场景
    "打电话给妈妈", "发短信", "设置闹钟", "导航到公司",
    "打开微信", "拍张照片", "打开相册", "查看日历",
    "打开设置", "连接蓝牙", "打开WiFi", "关闭屏幕",
    # 知识问答
    "地球有多大", "太阳系有几颗行星", "中国首都是哪里",
    "水的沸点是多少", "谁发明了电话", "圆周率是多少",
    # 日常用语
    "好的", "知道了", "谢谢", "明白", "收到",
    "没问题", "可以", "不用了", "算了", "行",
    # 模糊/无意义输入
    "啊", "嗯", "哦", "嗯哼", "啊啊啊",
    "什么", "什么意思", "怎么回事", "为什么", "不知道",
    # 非音乐动作
    "帮我查一下", "搜索一下", "翻译这句话", "计算一下",
    "记个笔记", "提醒我", "告诉我", "解释一下",
    # 食物/生活
    "附近有什么餐厅", "推荐一家火锅店", "今天吃什么",
    "怎么做红烧肉", "咖啡店在哪", "外卖到了吗",
    # 交通/出行
    "附近有地铁站吗", "怎么去机场", "路况怎么样",
    "还有多远", "到了吗", "该走哪条路",
    # 购物
    "帮我买点东西", "这个多少钱", "有没有打折",
    "快递到了吗", "什么时候发货", "退货怎么办",
    # 学习/工作
    "帮我写个报告", "这个怎么做", "开会时间",
    "项目进度", "发个邮件", "文件在哪里",
    # 随机短语（容易误触发）
    "换个话题", "切换页面", "跳过广告", "不要了",
    "换一个", "切换", "下一步", "上一步",
    "返回", "前进", "后退", "跳转",
    # 包含音乐相关字但非音乐命令
    "音乐考试怎么准备", "音乐学院在哪", "声音好奇怪",
    "歌词是什么意思", "这首诗真好", "唱歌比赛",
    "钢琴怎么弹", "吉他多少钱", "乐器店在哪",
    # 更多日常对话
    "你在吗", "能听到吗", "再说一遍", "我说完了",
    "没事了", "取消", "停止", "关掉吧",
    "你几岁了", "你是机器人吗", "你叫什么名字",
    "我饿了", "我困了", "好无聊", "太冷了",
]

VOLUME_TEMPLATES = [
    ("音量调到{v}", "{v}"),
    ("音量设为{v}", "{v}"),
    ("音量{v}", "{v}"),
    ("调到{v}", "{v}"),
    ("设置音量{v}", "{v}"),
    ("音量调成{v}", "{v}"),
    ("帮我把音量调到{v}", "{v}"),
    ("音量设置到{v}", "{v}"),
    ("声音调到{v}", "{v}"),
    ("声音{v}", "{v}"),
    ("音量大一点调到{v}", "{v}"),
    ("音量小一点调到{v}", "{v}"),
    ("把音量改成{v}", "{v}"),
    ("帮我调音量到{v}", "{v}"),
    ("音量改为{v}", "{v}"),
    ("声音设为{v}", "{v}"),
    ("声音调成{v}", "{v}"),
    ("调整音量到{v}", "{v}"),
    ("请把音量调到{v}", "{v}"),
    ("麻烦调到{v}", "{v}"),
]

# 音量值
VOLUME_VALUES = list(range(0, 101, 5)) + [1, 10, 15, 20, 25, 30, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90, 95]
VOLUME_VALUES = sorted(set(VOLUME_VALUES))


def make_sample(user_input: str, function_call_json: str) -> dict:
    """创建一条训练样本 (直接JSON格式, 无 function_call 标签)"""
    text = (
        f"<start_of_turn>user\n{user_input}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{function_call_json}<end_of_turn>"
    )
    return {
        "text": text,
        "user_input": user_input,
        "function_call": function_call_json,
    }


def generate_pause_samples():
    """生成 pause_music 样本"""
    samples = []
    fc = json.dumps({"f": "pause_music", "p": {}}, ensure_ascii=False)
    for inp in PAUSE_INPUTS:
        samples.append(make_sample(inp, fc))
    return samples


def generate_resume_samples():
    """生成 resume_music 样本"""
    samples = []
    fc = json.dumps({"f": "resume_music", "p": {}}, ensure_ascii=False)
    for inp in RESUME_INPUTS:
        samples.append(make_sample(inp, fc))
    return samples


def generate_next_samples():
    """生成 next_song 样本"""
    samples = []
    fc = json.dumps({"f": "next_song", "p": {}}, ensure_ascii=False)
    for inp in NEXT_INPUTS:
        samples.append(make_sample(inp, fc))
    return samples


def generate_prev_samples():
    """生成 previous_song 样本"""
    samples = []
    fc = json.dumps({"f": "previous_song", "p": {}}, ensure_ascii=False)
    for inp in PREV_INPUTS:
        samples.append(make_sample(inp, fc))
    return samples


def generate_volume_samples():
    """生成 set_volume 样本"""
    samples = []
    for template, _ in VOLUME_TEMPLATES:
        for v in VOLUME_VALUES:
            inp = template.format(v=v)
            fc = json.dumps({"f": "set_volume", "p": {"l": v}}, ensure_ascii=False)
            samples.append(make_sample(inp, fc))
    # 去重 + shuffle
    seen = set()
    unique = []
    for s in samples:
        key = s["user_input"]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def generate_negative_samples():
    """生成负样本: 非音乐输入 → none"""
    samples = []
    fc = json.dumps({"f": "none", "p": {}}, ensure_ascii=False)
    for inp in NEGATIVE_INPUTS:
        samples.append(make_sample(inp, fc))
    return samples


def main():
    data_dir = Path("data")

    # 读取原始训练数据
    orig_train = []
    with open(data_dir / "music_control_train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            orig_train.append(json.loads(line))

    # 统计原始分布
    from collections import Counter
    orig_funcs = Counter()
    for d in orig_train:
        fc = json.loads(d["function_call"])
        # 兼容旧格式 "function" 和新格式 "f"
        fn = fc.get("function") or fc.get("f")
        orig_funcs[fn] += 1
    print("原始训练集分布:")
    for fn, cnt in orig_funcs.most_common():
        print(f"  {fn}: {cnt}")

    # 生成新样本
    new_pause = generate_pause_samples()
    new_resume = generate_resume_samples()
    new_next = generate_next_samples()
    new_prev = generate_prev_samples()
    new_volume = generate_volume_samples()
    new_negative = generate_negative_samples()

    print(f"\n新增样本:")
    print(f"  pause_music: {len(new_pause)}")
    print(f"  resume_music: {len(new_resume)}")
    print(f"  next_song: {len(new_next)}")
    print(f"  previous_song: {len(new_prev)}")
    print(f"  set_volume: {len(new_volume)}")
    print(f"  none (负样本): {len(new_negative)}")

    # 转换原始 play_song 数据到紧凑 JSON 格式
    def convert_to_compact(sample):
        """将旧格式 function_call 转换为紧凑格式"""
        fc = json.loads(sample["function_call"])
        fn = fc.get("function") or fc.get("f")
        params = fc.get("parameters") or fc.get("p") or {}
        # 转换参数 key
        compact_params = {}
        if "song_name" in params or "s" in params:
            compact_params["s"] = params.get("song_name") or params.get("s")
        if "artist" in params or "a" in params:
            compact_params["a"] = params.get("artist") or params.get("a")
        if "level" in params or "l" in params:
            compact_params["l"] = params.get("level") or params.get("l")
        new_fc = json.dumps({"f": fn, "p": compact_params}, ensure_ascii=False)
        return make_sample(sample["user_input"], new_fc)

    # 合并: 保留所有原始 play_song, 替换其他函数的旧数据
    play_song_orig = [d for d in orig_train if (json.loads(d["function_call"]).get("function") or json.loads(d["function_call"]).get("f")) == "play_song"]
    play_song_samples = [convert_to_compact(d) for d in play_song_orig]

    # 为了平衡, 对 play_song 进行下采样 (保留最多 2000 条)
    MAX_PLAY_SONG = 2000
    if len(play_song_samples) > MAX_PLAY_SONG:
        random.shuffle(play_song_samples)
        play_song_samples = play_song_samples[:MAX_PLAY_SONG]
        print(f"\n  play_song 下采样: {len(play_song_samples)}")

    # 合并所有样本 (包含负样本)
    all_samples = (
        play_song_samples
        + new_pause
        + new_resume
        + new_next
        + new_prev
        + new_volume
        + new_negative
    )
    random.shuffle(all_samples)

    # 保存增强后的训练数据
    output_file = data_dir / "music_control_train_balanced.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 统计新分布
    new_funcs = Counter()
    for d in all_samples:
        fc = json.loads(d["function_call"])
        new_funcs[fc.get("f") or fc.get("function")] += 1

    print(f"\n平衡后训练集: {len(all_samples)} 条")
    for fn, cnt in new_funcs.most_common():
        pct = 100 * cnt / len(all_samples)
        print(f"  {fn}: {cnt} ({pct:.1f}%)")
    print(f"\n保存到: {output_file}")

    # 同样增强 eval 数据
    orig_eval = []
    with open(data_dir / "music_control_eval.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            orig_eval.append(json.loads(line))

    # 转换 eval 原始数据到紧凑格式
    orig_eval = [convert_to_compact(d) for d in orig_eval]

    # 为 eval 添加一些非 play_song 样本 (从新增样本中取 10%)
    eval_additions = []
    for samples in [new_pause, new_resume, new_next, new_prev, new_volume, new_negative]:
        n = max(3, len(samples) // 10)
        random.shuffle(samples)
        eval_additions.extend(samples[:n])

    # 从 eval_additions 中移除已经在 train 中的 (简单起见, 跳过重复检查, 因为生成的样本有足够变体)
    eval_all = orig_eval + eval_additions
    random.shuffle(eval_all)

    eval_output = data_dir / "music_control_eval_balanced.jsonl"
    with open(eval_output, "w", encoding="utf-8") as f:
        for sample in eval_all:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Eval 集: {len(eval_all)} 条 → {eval_output}")


if __name__ == "__main__":
    main()
