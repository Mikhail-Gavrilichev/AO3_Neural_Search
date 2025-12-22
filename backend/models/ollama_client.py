import ollama
from typing import List, Dict


class OllamaExplainer:
    def __init__(self, model_name: str):
        self.model = model_name

    def build_prompt(
        self,
        query_text: str,
        retrieved_works: List[Dict],
        include_full_text: bool
    ) -> str:
        works_text = []

        for i, work in enumerate(retrieved_works, 1):
            if include_full_text:
                works_text.append(
                    f"""
                    Работа {i}:
                    Название: {work.get('title')}
                    Краткое описание: {work.get('summary')}
                    Теги: {', '.join(work.get('tags', []))}
                    Сходство: {work.get('similarity_score')}
                    Содержание: {work.get('text_sample')}
                    """.strip()
                )
            else:
                works_text.append(
                    f"""
                    Работа {i}:
                    Название: {work.get('title')}
                    Краткое описание: {work.get('summary')}
                    Теги: {', '.join(work.get('tags', []))}
                    Сходство: {work.get('similarity_score')}
                    """.strip())

        return f"""
# ЗАДАЧА: АНАЛИЗ РЕЗУЛЬТАТОВ ЛИТЕРАТУРНОГО ПОИСКА

## ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
"{query_text}"

## НАЙДЕННЫЕ ПРОИЗВЕДЕНИЯ:
{chr(10).join(works_text)}

## ИНСТРУКЦИЯ ДЛЯ АНАЛИТИКА:

Ты — эксперт по литературному анализу и рекомендательным системам.

### ШАГ 1: Проверь, нет ли заблокированных работ
- Если видишь "Работа заблокирована" в любом поле → ответь: "Простите, но затрагиваемые данной работой темы запрещены."

### ШАГ 2: Если все работы разрешены, проведи анализ:

1. **ПРИЧИНЫ НАХОДКИ** — почему поисковая система выбрала именно эти работы?
   - Совпадение по ключевым словам/темам
   - Стилистическое сходство
   - Тематическое соответствие

2. **ОБЩИЕ ТЕМЫ И ТЕГИ** — что объединяет запрос и найденные работы?
   - Перечисли конкретные теги из работ, которые релевантны запросу
   - Укажите тематические пересечения

3. **КРАТКИЙ ВЫВОД** (3-5 предложений):
   - Общая характеристика найденных произведений
   - Насколько они соответствуют запросу
   - Какие аспекты запроса лучше всего отражены

### ТРЕБОВАНИЯ:
- Отвечай только на русском языке
- Будь конкретен, опирайся только на предоставленные данные
- Не выдумывай несуществующие детали
- Используй цифры и факты из данных (сходство, теги и т.д.)
- Формат: ясный, структурированный, без лишних слов

Начинай анализ:
"""

    def explain(self, query_text: str, retrieved_works: List[Dict], include_full_text: bool) -> str:
        prompt = self.build_prompt(query_text, retrieved_works, include_full_text)

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]

