import streamlit as st
import requests
import json
from typing import Dict

st.set_page_config(
    page_title="Neural Search для литературных произведений",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"


def check_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            return True
    except:
        pass
    return False


def search_similar_works(title: str, summary: str, fandoms: str, characters: str, story: str, top_k: int = 5, llm_mode: str = ""):
    try:
        payload = {
            "title": title,
            "summary": summary,
            "fandoms": fandoms,
            "characters": characters,
            "story": story,
            "top_k": top_k,
            "llm_mode": llm_mode
        }

        response = requests.post(f"{API_BASE_URL}/search", json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка соединения: {str(e)}")
        return None


def add_new_work(work_data: Dict):
    try:
        response = requests.post(f"{API_BASE_URL}/add-work", json=work_data)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка соединения: {str(e)}")
        return None


def main():
    st.title("Neural Search для литературных произведений")
    st.markdown("---")

    if not check_api_connection():
        st.error("Нет соединения с сервером API.")
        if st.button("Попробовать снова"):
            st.rerun()
        return

    st.success("Соединение с API установлено")

    with st.sidebar:
        st.header("Навигация")
        page = st.radio(
            "Выберите раздел:",
            ["Поиск произведений", "Добавить новую работу", "Разработчики"]
        )

        st.markdown("---")
        st.header("Информация")

        try:
            info_response = requests.get(f"{API_BASE_URL}/dataset-info")
            if info_response.status_code == 200:
                info = info_response.json()
                st.info(f"В базе: {info['total_works']} работ")
                st.info(f"Тегов: {info['total_tags']}")
        except:
            pass

        st.markdown("---")
        st.markdown("""
        **Инструкция:**
        1. Заполните информацию о произведении
        2. Нажмите "Найти похожие работы"
        3. Просмотрите результаты поиска
        """)

    if page == "Поиск произведений":
        show_search_page()
    elif page == "Добавить новую работу":
        show_add_work_page()
    elif page == "Разработчики":
        show_developers_page()


def show_search_page():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Информация о произведении")

        title = st.text_input("Название произведения*",
                              placeholder="Введите название...")

        summary = st.text_area("Краткое описание (SUMMARY)",
                               placeholder="Опишите сюжет...",
                               height=100)

        fandoms = st.text_input("Фэндомы (через запятую)",
                                placeholder="Например: Гарри Поттер, Марвел...")

        characters = st.text_input("Персонажи (через запятую)",
                                   placeholder="Например: Гарри Поттер, Гермиона Грейнджер...")
        st.subheader("Использование модели анализа")

        llm_mode = st.radio(
            "Режим работы DeepSeek:",
            options=[
                ("Не использовать модель", "off"),
                ("Только для моей работы", "query_only"),
                ("Для всех работ", "full")
            ],
            format_func=lambda x: x[0]
        )[1]

    with col2:
        st.subheader("Настройки поиска")

        top_k = st.slider("Количество результатов", 1, 10, 5)

        st.markdown("---")

    st.subheader("Текст произведения (STORY)")
    story = st.text_area(
        "Введите текст произведения или отрывок",
        placeholder="Начните писать здесь...",
        height=200,
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button(
            "Найти похожие работы",
            type="primary",
            use_container_width=True
        )

    if search_button:
        if not title and not story:
            st.warning("Пожалуйста, заполните хотя бы название или текст произведения")
            return

        with st.spinner("Ищем похожие работы..."):
            result = search_similar_works(title, summary, fandoms, characters, story, top_k, llm_mode)

        if result:
            display_search_results(result)


def display_search_results(result: Dict):
    st.markdown("---")
    st.subheader(f"Найдено {result['results_count']} похожих работ")

    if result.get("llm_explanation"):
        st.markdown("Анализ модели")
        st.info(result["llm_explanation"])
    elif result.get("llm_mode") == "off":
        st.info("Анализ модели отключён")

    with st.expander("Информация о вашем запросе"):
        query_info = result['query']
        cols = st.columns(4)
        cols[0].metric("Название", query_info['title'] or "Не указано")
        cols[1].metric("Длина описания", f"{len(query_info['summary'] or '')} симв.")
        cols[2].metric("Фэндомы", query_info['fandoms'] or "Не указаны")
        cols[3].metric("Персонажи", query_info['characters'] or "Не указаны")

    for i, work in enumerate(result['results']):
        with st.container():
            st.markdown(f"### {i + 1}. {work['title']}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Общий скоринг", f"{work['similarity_score']:.3f}")
            col2.metric("Косинусное сходство", f"{work['cosine_similarity']:.3f}")
            col3.metric("Совпадение тегов", work['tag_overlap'])
            col4.metric("ID работы", work['work_id'])

            with st.expander("Подробная информация"):
                tab1, tab2, tab3, tab4 = st.tabs(["Описание", "Теги", "Персонажи", "Ссылки"])

                with tab1:
                    if work['summary']:
                        st.write(work['summary'])
                    else:
                        st.info("Описание отсутствует")

                with tab2:
                    if work['tags']:
                        tags_html = " ".join([
                                                 f"<span style='background-color: #e0f7fa; padding: 4px 8px; margin: 2px; border-radius: 4px; display: inline-block;'>{tag}</span>"
                                                 for tag in work['tags'][:20]])
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.info("Теги отсутствуют")

                with tab3:
                    if work['characters']:
                        st.write(", ".join(work['characters']))
                    else:
                        st.info("Персонажи не указаны")

                with tab4:
                    if work['work_id']:
                        st.write(f"Work ID: {work['work_id']}")
                        st.markdown(f"[Открыть работу](https://archiveofourown.org/works/{work['work_id']})")

            st.markdown("---")


def show_add_work_page():
    st.title("Добавить новую работу в базу")
    st.markdown("---")

    with st.form("add_work_form"):
        st.subheader("Основная информация")

        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("Название произведения*",
                                  placeholder="Обязательное поле")
            summary = st.text_area("Краткое описание",
                                   placeholder="Опишите сюжет...",
                                   height=150)
            fandoms = st.text_area("Фэндомы (по одному в строке)",
                                   placeholder="Например:\nГарри Поттер\nМарвел",
                                   height=100)

        with col2:
            characters = st.text_area("Персонажи (по одному в строке)",
                                      placeholder="Например:\nГарри Поттер\nГермиона Грейнджер",
                                      height=100)
            categories = st.text_area("Категории (по одному в строке)",
                                      placeholder="Например:\nGen\nF/M",
                                      height=80)
            relationships = st.text_area("Отношения/пейринги (по одному в строке)",
                                         placeholder="Например:\nГарри Поттер/Гермиона Грейнджер",
                                         height=80)

        st.subheader("Теги и содержание")

        tags = st.text_area("Теги (через запятую)",
                            placeholder="Например: romance, adventure, fantasy, дружба",
                            height=80)

        text_sample = st.text_area("Текст произведения или отрывок*",
                                   placeholder="Введите текст произведения...",
                                   height=200)

        work_id = st.text_input("ID работы (AO3)",
                                placeholder="Например: 17231411"
        )

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "Добавить работу",
                type="primary",
                use_container_width=True
            )

        if submitted:
            if not title or not text_sample:
                st.error("Заполните обязательные поля (название и текст)")
                return

            work_data = {
                "work_id": work_id,
                "title": title,
                "summary": summary if summary else "",
                "tags": [tag.strip() for tag in tags.split(",")] if tags else [],
                "fandoms": [f.strip() for f in fandoms.split("\n") if f.strip()] if fandoms else [],
                "characters": [c.strip() for c in characters.split("\n") if c.strip()] if characters else [],
                "categories": [cat.strip() for cat in categories.split("\n") if cat.strip()] if categories else [],
                "relationships": [rel.strip() for rel in relationships.split("\n") if
                                  rel.strip()] if relationships else [],
                "text_sample": text_sample
            }

            with st.spinner("Добавляем работу в базу..."):
                result = add_new_work(work_data)

            if result and result.get("success"):
                st.success(f"Работа успешно добавлена! ID: {result['work_id']}")

                with st.expander("Просмотр добавленной работы"):
                    st.json(work_data)
            else:
                st.error("Ошибка при добавлении работы")


def show_developers_page():
    st.title("Команда разработчиков")
    st.markdown("---")

    developers = [
        {
            "name": "Гавриличев Михаил Алексеевич",
            "group": "ФИТ-231",
            "email": "mikhail05@list.ru",
            "role": "Тимлид, разработчик модели"
        },
        {
            "name": "Кокорин Артём Владимирович",
            "group": "ФИТ-231",
            "email": "9proger@list.ru",
            "role": "Разработчик бэкенда"
        },
        {
            "name": "Ильин Максим Викторович",
            "group": "ФИТ-231",
            "email": "Prokrastinator2000@list.ru",
            "role": "Разработчик фронтенда"
        }

    ]

    cols = st.columns(3)

    for idx, dev in enumerate(developers):
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"""
                <div style='
                    background-color: #f0f2f6;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                    text-align: center;
                '>
                    <h3>{dev['name']}</h3>
                    <p><strong>Роль:</strong> {dev['role']}</p>
                    <p><strong>Группа:</strong> {dev['group']}</p>
                    <p><strong>Email:</strong> {dev['email']}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("О проекте")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Технологии:**
        - **Модель**: BigBird (RoBERTa-base)
        - **Бэкенд**: FastAPI + PyTorch
        - **Фронтенд**: Streamlit
        - **Векторный поиск**: Cosine Similarity

        **Основные функции:**
        - Нейронный поиск по семантике текста
        - Автоматическое присвоение тегов
        - Поиск по косинусному сходству
        - Расширяемая база данных
        """)

    with col2:
        st.markdown("""
        **Архитектура системы:**
        1. **Клиент** (Streamlit) - Интерфейс пользователя
        2. **API** (FastAPI) - Обработка запросов
        3. **ML модель** (BigBird) - Эмбеддинги и классификация
        4. **Векторная БД** - Хранение эмбеддингов
        5. **Dataset** - Литературные произведения

        **Особенности:**
        - Поддержка длинных текстов (до 4096 токенов)
        - Многометочная классификация
        - Инкрементальное обучение
        - RESTful API
        - Интеграция с DeepSeek для семантического поиска
        """)

    st.markdown("---")

    st.subheader("Контактная информация")

    contact_cols = st.columns(3)
    contact_cols[0].info("**GitHub репозиторий:**\ngithub.com/Mikhail-Gavrilichev/AO3_Neural_Search")


if __name__ == "__main__":
    main()