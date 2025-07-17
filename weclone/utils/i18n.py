from typing import Dict, List, Optional


class MultiLangList:
    def __init__(self, translations: Dict[str, List[str]], default_lang="en"):
        self.translations = translations
        self.current_lang = default_lang
        self.default_lang = default_lang
        # Validate that all translation lists have the same length
        self._validate_translations()
        # 创建反向映射字典，用于快速查找
        self._build_reverse_mapping()

    def _validate_translations(self):
        """Validate that all translation lists have the same length"""
        if not self.translations:
            raise ValueError("Translations dictionary cannot be empty")

        # Get the length of the first list as reference
        first_lang = next(iter(self.translations))
        expected_length = len(self.translations[first_lang])

        # Check if all lists have the same length
        for lang, items in self.translations.items():
            if len(items) != expected_length:
                raise ValueError(
                    f"Translation list for '{lang}' has {len(items)} items, "
                    f"expected {expected_length} items (same as '{first_lang}')"
                )

    def _build_reverse_mapping(self):
        """构建反向映射，用于根据文本查找对应的索引和其他语言翻译"""
        self.text_to_index = {}  # 文本 -> (语言, 索引)

        for lang, items in self.translations.items():
            for index, text in enumerate(items):
                self.text_to_index[text.lower()] = (lang, index)

    def set_language(self, lang: str):
        """设置当前语言"""
        if lang in self.translations:
            self.current_lang = lang
            return self
        else:
            print(f"Warning: Language '{lang}' not available, using default")

    def get_items(self, lang: Optional[str] = None) -> List[str]:
        """获取指定语言的列表"""
        target_lang = lang or self.current_lang
        return self.translations.get(target_lang, self.translations[self.default_lang])

    def get_item(self, index: int, lang: Optional[str] = None) -> str:
        """获取指定索引的翻译项"""
        items = self.get_items(lang)
        if 0 <= index < len(items):
            return items[index]
        raise IndexError("List index out of range")

    def translate_text(self, text: str, target_lang: Optional[str] = None) -> Optional[str]:
        """
        根据输入的文本（中文或英文）获取另一种语言的翻译

        Args:
            text: 要翻译的文本
            target_lang: 目标语言，如果不指定则自动判断（中文->英文，英文->中文）

        Returns:
            翻译后的文本，如果找不到则返回None
        """
        text_lower = text.lower()

        # 查找文本在哪个语言的哪个位置
        if text_lower not in self.text_to_index:
            return None

        source_lang, index = self.text_to_index[text_lower]

        # 如果没有指定目标语言，则自动判断
        if target_lang is None:
            if source_lang == "en":
                target_lang = "zh_CN"  # 英文->中文
            elif source_lang == "zh_CN":
                target_lang = "en"  # 中文->英文
            else:
                return None

        # 获取目标语言的翻译
        if target_lang in self.translations:
            target_items = self.translations[target_lang]
            if index < len(target_items):
                return target_items[index]

        return None

    def get_translation_pair(self, text: str) -> Dict[str, str]:
        """
        获取某个文本的中英文对照

        Args:
            text: 要查找的文本

        Returns:
            包含中英文翻译的字典，例如 {'en': 'Administrator', 'zh_CN': '管理员'}
        """
        text_lower = text.lower()

        if text_lower not in self.text_to_index:
            return {}

        source_lang, index = self.text_to_index[text_lower]

        result = {}
        for lang in ["en", "zh_CN"]:
            if lang in self.translations and index < len(self.translations[lang]):
                result[lang] = self.translations[lang][index]

        return result

    def translate_batch(self, texts: List[str], target_lang: Optional[str] = None) -> List[Optional[str]]:
        """
        批量翻译文本

        Args:
            texts: 要翻译的文本列表
            target_lang: 目标语言

        Returns:
            翻译结果列表
        """
        return [self.translate_text(text, target_lang) for text in texts]

    def __iter__(self):
        return iter(self.get_items())

    def __len__(self):
        return len(self.get_items())

    def __getitem__(self, index):
        return self.get_item(index)


if __name__ == "__main__":
    # 定义中英文双语数据
    user_types_data = {
        "en": ["Administrator", "Regular User", "Guest", "Moderator", "Super Admin"],
        "zh_CN": ["管理员", "普通用户", "访客", "版主", "超级管理员"],
    }

    status_data = {
        "en": ["Active", "Inactive", "Pending", "Suspended", "Deleted"],
        "zh_CN": ["活跃", "非活跃", "待定", "暂停", "已删除"],
    }

    permission_data = {
        "en": ["Read", "Write", "Execute", "Delete", "Admin"],
        "zh_CN": ["读取", "写入", "执行", "删除", "管理"],
    }

    # 创建多语言列表
    user_types = MultiLangList(user_types_data)
    status_list = MultiLangList(status_data)
    permissions = MultiLangList(permission_data)
    # 使用示例
    print("=== 基本翻译功能 ===")
    # 中文翻译为英文
    result1 = user_types.translate_text("管理员")
    print(f"'管理员' -> '{result1}'")  # 输出: '管理员' -> 'Administrator'

    # 英文翻译为中文
    result2 = user_types.translate_text("Guest")
    print(f"'Guest' -> '{result2}'")  # 输出: 'Guest' -> '访客'

    # 指定目标语言
    result3 = user_types.translate_text("管理员", target_lang="en")
    print(f"'管理员' -> '{result3}' (指定英文)")  # 输出: '管理员' -> 'Administrator' (指定英文)

    print("\n=== 获取中英文对照 ===")
    translation_pair = user_types.get_translation_pair("Administrator")
    print(f"'Administrator' 的中英文对照: {translation_pair}")
    # 输出: {'en': 'Administrator', 'zh_CN': '管理员'}

    print("\n=== 批量翻译 ===")
    chinese_texts = ["管理员", "普通用户", "访客"]
    english_results = user_types.translate_batch(chinese_texts)
    print(f"批量翻译结果: {list(zip(chinese_texts, english_results))}")
    # 输出: [('管理员', 'Administrator'), ('普通用户', 'Regular User'), ('访客', 'Guest')]

    print("\n=== 状态列表翻译 ===")
    status_result = status_list.translate_text("活跃")
    print(f"'活跃' -> '{status_result}'")  # 输出: '活跃' -> 'Active'

    status_result2 = status_list.translate_text("Pending")
    print(f"'Pending' -> '{status_result2}'")  # 输出: 'Pending' -> '待定'

    print("\n=== 权限翻译 ===")
    perm_result = permissions.translate_text("读取")
    print(f"'读取' -> '{perm_result}'")  # 输出: '读取' -> 'Read'

    print("\n=== 错误处理 ===")
    not_found = user_types.translate_text("不存在的文本")
    print(f"不存在的文本翻译结果: {not_found}")  # 输出: None

    print("\n=== 当前语言设置 ===")
    user_types.set_language("zh_CN")
    print(f"当前语言列表: {list(user_types)}")  # 输出中文列表

    user_types.set_language("en")
    print(f"切换后列表: {list(user_types)}")  # 输出英文列表
