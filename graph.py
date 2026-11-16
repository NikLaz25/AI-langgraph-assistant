"""
Модуль: backend/app/graph.py
Версия: 0.1.1
Назначение: граф ассистента
"""

# Библиотеки
#=========================================
from backend.app.config import config

from dataclasses import dataclass
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool


from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models # набор pydantic‑моделей, описывающих запросы и ответы API Qdrant
from functools import partial


# КЛИЕНТ LLM
#=========================================
client_llm = init_chat_model(
    model = config.llm,
    base_url = config.url,
    api_key = config.local_api_key,
    temperature = config.default_temperature,
    model_provider="openai",
)

## Создаём нужные структуры данных, обертки типов
#================================================
@dataclass(frozen=True)
class Context:
    user_id: str = Field(description = "уникальный id пользователя")

@dataclass(frozen=True)
class AssistState(MessagesState):
    rag_docs: list[dict] = Field(description="Список релевантных чанков")
    next_node: Literal["what_find", "call_llm", "off_topic"] = Field(description="Следующая нода графа")
    what_find_list: dict = Field(description="список перефразированных коротких вопросов для поиска в RAG")
    where_find_dict: dict = Field(description="словарь с перечнем тем для поискового фильтра к каждому вопросу из списка what_find_list")
    security_blocked: bool = Field(
        default=False,
        description="Признак попытки взлома ассистента. Запрос заблокирован фильтром безопасности"
    )

## Нода безопасность
#===============================================
def security(state: AssistState)-> dict:
    """Отвечает за безопасность
    Args:
        state:
            текущее состояние графа
    Returns:
        dict: обновленное состояние графа с новым сообщением модели
    Сложность О большое: O(n)
    """

    secure_list = [
        "украина", "президент", "политика", "крым", "америка", "войн"
    ]

    user_question = state["messages"][-1].content

    for i_word in secure_list:
        if i_word in user_question.lower():
            return {
                "security_blocked": True,
                "messages": [AIMessage(content="Запрещенный запрос! Сработала блокировка")]
            }

    return {
        "security_blocked": False
    }


## Нода Оркестратор
#==============================================
class OrchStruct(BaseModel):
    """Описывает выбранный маршрут обработки зазапроса
    Returns:
    Применяется в функции оркестратора для выбора сценария обработки запроса пользователя
        """
    next_node: Literal["what_find", "call_llm", "off_topic"] = Field(description="Следующая нода графа")

# специально для ноды оркестратора создаём структурированный клиент LLM
client_orch_struct = client_llm.with_structured_output(OrchStruct)

def orchestrator(state: AssistState)-> dict:
    """Принимает решение по какому сценарию двигаться

    Args:
        state: текущее состояние графа, содержащее вопрос пользователя
    Returns:
        dict: обновление состояния графа.

    Сложность О большое: О(1)
    """

    if state.get('security_blocked', False):
        return {"next_node": END}


    # системный промпт
    system_prompt = SystemMessage(
        content=f"""
        # Темы аналитического управления:
        1. система статусов
        2. статусы
        3. отчетность
        4. отчётность аналитического управления
        5. прогнозирование спроса
        6. модель рентабельности
        7. рентабельность
        8. перемещение
        9. перемещение продукции
        10. перемещение продукции между складами
        11. перемещения
        12. управление запасами
        13. управление запасами продукции на складах
        14. доставка
        15. ускорение
        16. ускорение доставки

        # Ты оркестратор корпоративного аналитического ассистента
        1. Если вопрос хоть немного связан хотя бы с одной из тем аналитического управления или уточнение по данной теме, выбери ноду 'what_find'.
        2. Если в истории диалога уже есть данные для ответа, выбери ноду 'call_llm'.
        3. Если вопрос не касается тем аналитического управления или уточнения по ним, выбери ноду 'off_topic'
        """
    )

    result = client_orch_struct.invoke(
        [system_prompt,
        *state['messages'],
         ]
    )

    return {"next_node": result.next_node}

## Нода Off Topic
#===============================================
def off_topic():
    """
    Стандартный отказ пользователю, по причине нетематического вопроса
    Return: dict
    """
    return  {"messages": ["Задайте вопрос по теме связанной с аналитическим управлением!"]}

## Нода Что искать? Переформулировки вопроса
#===============================================
class StructWhatFind(BaseModel):
    '''Класс описывает структуру ответов ноды (Что искать)'''
    rephrased_question_list: list[str] = Field(
        description=('Список перефразированных вопросов')
    )

# создадим клиент со структурированным выходом
client_what_struct = client_llm.with_structured_output(StructWhatFind)

def call_what_find(
        state: MessagesState,
):
    """
    Обрабатывает вопрос пользователя, готовит список коротких вопросов для поиска
    Args:
        state: текущее состояние графа, содержащее вопрос пользователя
    Returns:
        dict: обновление состояния графа.

    Сложность О большое: О(1)
    """
    # подготовим системный промпт
    system_prompt = SystemMessage(
        content = ("Ты профессиональный помощник для поиска в векторной базе данных."
                  "Твоя задача переформулировать сообщение пользователя, и выдать не более пяти формулировок запроса пользователя для правильного поиска в векторной базе данных."
                  "При необходимости можешь разбивать вопрос пользователя на логические части."
                  "Всё что не относится к основному смыслу вопроса нужно убрать"
                  "Ответ должен быть в структурированном виде. Структура - список обработанных вопросов."
                  "Например: "
                  "Вопрос пользователя: Привет дружище, расскажи ка мне что такое рентабельность будущих периодов. А и ещё забыл, а чтотакое перемещение продукции, о чем тут речь?"
                  "Ответ: ['рентабельность', 'рентабельность будущих периодов', 'перемещение продукции']"
                   )
    )
    # получим актуальный опрос пользователя
    user_question = state["messages"][-1].content

    print(user_question)

    # направим запрос в LLM, получим ответ response
    response_list = client_what_struct.invoke(
        [system_prompt, 'user_question:'+user_question],
    )

    return {
        "what_find_list": response_list.rephrased_question_list # rephrased_question_list для упрощения структуры
    }

## Нода Где искать?
#===============================================
QDRANT_PATH = (
    r"C:\Users\lazarevnv\Desktop\repository_for_clone"
    r"\au_for_testing\tmp\local_qdrant"
)

COLLECTION_NAME = "au_docs_collection"

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=config.openai_api_key,
)

#### QdrantClient подключаемся к базе
client = QdrantClient(path=QDRANT_PATH)

#### vector_store, QdrantVectorStore

# Подключаем LangChain к существующей коллекции.
# Новые документы здесь не добавляются.
# vector_store - это экземпляр для работы с базой данных
vector_store = QdrantVectorStore(
    client=client, # qdrant_client
    collection_name=COLLECTION_NAME, # имя коллекции
    embedding=embeddings_model, # модель векторных представлений
)

#### get_all_topics вспомогательная функция

# Это не создаёт новые данные и не переиндексирует embeddings (векторные представления). Это индекс для быстрого поиска и фильтрации по метаданным

client.create_payload_index( #  создаёт быстрый индекс по полю метаданных (metadata.topics), чтобы потом можно было быстро фильтровать документы по темам.
    collection_name="au_docs_collection",
    field_name="metadata.topics",
    field_schema=models.PayloadSchemaType.KEYWORD, # сообщает Qdrant, что поле содержит ключевые слова (строки без токенизации), т.е. поиск будет работать подобно обычному индексу в SQL
)

def get_all_topics(
        client: QdrantClient,
        collection_name: str,
)-> list[str]:
    """
    Получает список всех уникальных topics в указанной коллекции базы данных Qdrant

    Args:
        client: клиент для подключения к Qdrant.
        collection_name: наименование коллекции Qdrant.
    Returns:
        list[str]: список уникальных значений поля metadata.topics.

    Сложность O большое: О(n).
    """

    # получаем уникальные значения поля metadata.topics
    result = client.facet(
        collection_name=collection_name,
        key="metadata.topics",
        limit=100,
        exact=True,
    )

    # извлекаем только значения topics
    topics_list = [hit.value for hit in result.hits]

    return topics_list

class WhereFindStruct(BaseModel):
    """Класс описывает структуру ответов ноды (ГДЕ искать)"""
    target_topics_list: list[str] = Field(
        description = ('список целевых topics для фильтра поиска в базе данных')
    )

# создадим клиент со структурированным выходом
client_where_struct = client_llm.with_structured_output(WhereFindStruct)

def where_find_call(
        state: AssistState,
        full_topics_list,
):
    """Функция поиска topics для каждого обработанного вопроса
    Args:
        state: текущее состояние графа со списком обработанных вопросов what_find_list
        full_topics_list: список всех тем в используемой базе данных
    Returns:
        dict: обновление состояния с topics_list по каждому вопросу

    Сложность O большое: О(n).
    """
    # получаем список what_find_list
    # rephrased_question_list больше не нужен в новой упрощенной структуре
    what_find_list = state.get('what_find_list')

    # системный промпт
    system_prompt = SystemMessage(
        content = ("Ты профессиональный помощник для формирования фильтра поиска в векторной базе данных."
                   "Твоя задача выбрать из перечня topics только те, которые касаются темы целевого вопроса target_question."
                   f"Перечень topics : {full_topics_list}"
                   "Требуется структурированный ответ в виде списка"
                   )
    )

    # создаём пустой список
    result_dict = {}


    # Циклом обрабатываем список what_find_list
    for target_question in what_find_list:

        topics_list = client_where_struct.invoke(
            [system_prompt, "target_question: "+target_question]
        )

        # добавляем ответ в словарь
        result_dict[target_question] = topics_list.target_topics_list
        # присваиваем значение именно target_topics_list как подраздела, это для простой и понятной структуры

    return {
        "where_find_dict" : result_dict
    }


## Нода retriever
#===============================================

### Вспомогательная функция для списка уникальных чанков
def chunks_uniq(
        chunks_list: list,
):
    """
    Удаляет дубликаты чанков в списке
    Args:
        chunks_list: список чанков
    Returns:
        uniq_chunks_list: список уникальных чанков

    Сложность О большое:
        О(n)
        """
    # сформируем пустой список уникальных чанков для заполнения
    uniq_chunks_list = []
    # сформируем множество для фиксирования chunk_id
    set_chunk_id = set()

    # циклом пройдём по каждому элементу списка
    for i_chunk in chunks_list:

        # получаем id чанка
        chunk_id = i_chunk.metadata['chunk_id']

        # если chunk_id нет в множестве set_chunk_id, то добавляем этот чанк в uniq_chunks_list
        if chunk_id not in set_chunk_id:
            uniq_chunks_list.append(i_chunk)

        # добавляем в множество chunk_id для последующей проверки на дубликаты
        set_chunk_id.add(chunk_id)

    return uniq_chunks_list


def retriever(
        state: AssistState,
):
    """ Выполняет семантический поиск по каждому обработанному вопросу из списка
    и возвращает общий список уникальных найденных чанков.

    Args:
        state: текущее состояние графа, содержащее словарь where_find_dict
            с вопросами и соответствующими им темами поиска.

    Returns:
        dict: обновление состояния графа со списком уникальных найденных документов
        в поле 'rag_docs'.

    Сложность О большое:
        О(n*k), где n - количество поисковых вопросов,
        k - число найденных чанков на один вопрос.
    """
    # Получим словарь для поиска чанков
    search_dict = state.get('where_find_dict')

    # Задаём пустой список для найденных документов
    found_docs_list = list()

    # Цикл по вопросам
    for question, topics_list in search_dict.items():
        # получаем список тем для конкретного вопроса
        # topics.target_topics_list выделять уже не нужно, это сделано в ноде where_find
        # topics_list = topics.target_topics_list

        # формируем фильтр
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.topics",
                    match=models.MatchAny(
                        any=topics_list,
                    ),
                )
            ]
        )

        # семантический поиск в базе данных
        chunks_list = vector_store.similarity_search(
            query=question,
            k=3,
            filter=search_filter)

        # Добавляем найденные чанки в общее множество уникальных чанков
        found_docs_list.extend(chunks_list)

    # оставляем только уникальные чанки
    unik_chunks_list = chunks_uniq(found_docs_list)

    # возвращаем общий список уникальных чанков в нужном формате для добавления в состояние графа
    return {'rag_docs': unik_chunks_list}

# Нода call_llm
#===============================================
@tool
def calculate(a: int, b: int)-> int:
    """Суммирует два числа"""
    return a + b

tools_list = [calculate,]

# клиент llm с инструментами
client_tools = client_llm.bind_tools(tools_list)



def call_model(
        state: AssistState,
        runtime: Runtime[Context],
)-> dict:
    """Вызывает языковую модель

    Args:
        state:
            текущее состояние графа с историей сообщений и данными RAG
        runtime:
            среда выполнения графа с долговременной памятью - данными о пользователе

    Returns:
        dict: обновленное состояние графа с новым сообщением модели

    Применяется при сборке графа
    Сложность О большое:
    """

    # получаем данные о пользователе из долговременной памяти
    user_id = runtime.context.user_id
    namespace = (user_id, 'profile')
    profile = runtime.store.get(namespace,"main",)
    if profile:
        user_info = profile.value
    else:
        user_info = {}

    # долучаем данные RAG
    rag_docs_list = state.get("rag_docs") or []
    rag_info= "\n\n".join(i_doc.page_content for i_doc in rag_docs_list)

    # формируем систенмное сообщение для LLM
    system_prompt = SystemMessage(
        content=(
            f"""Ты являешься профессиональным консультантом
            Аналитического управления IEK GROUP.\n
            Отвечай на вопрос только на основании найденной
            информации.\n
            Если информации недостаточно, сообщи об этом.\n
            У тебя есть инструменты, при необходимости ты можешь их вызвать.\n
            Информация о пользователе:\n{user_info}\n
            Найденная информация:\n{rag_info}"""
        )
    )

    messages = [
        system_prompt,
        *state["messages"]
    ]

    # вызываем LLM
    response = client_tools.invoke(messages)

    return {"messages": [response]}

## Нода TOOLS
#===============================================
tool_node = ToolNode(tools_list)

# получаем список тем, это категории вопросов
full_topics_list = get_all_topics(
    client=client,
    collection_name="au_docs_collection"
)


## Собираем граф
#===============================================
#Создаем конструктор графа
builder_for_graph =  StateGraph(
    AssistState,
    context_schema=Context,
)


# добавляем ноды
builder_for_graph.add_node(
    "security",
    security,
)

builder_for_graph.add_node(
    "orchestrator",
    orchestrator,
)

builder_for_graph.add_node(
    "off_topic",
    off_topic,
)

builder_for_graph.add_node(
    "what_find",
    call_what_find,
)

builder_for_graph.add_node(
    "where_find",
    partial(
        where_find_call,
        full_topics_list = full_topics_list,
    ),
)

builder_for_graph.add_node(
    "retriever",
    retriever,
)

builder_for_graph.add_node(
    "call_llm",
    call_model,
)

builder_for_graph.add_node(
    "tools",
    tool_node,
)


# добавляем связи в граф
builder_for_graph.add_edge(START, "security")

builder_for_graph.add_edge("security", "orchestrator")

builder_for_graph.add_conditional_edges(
    "orchestrator",
    lambda state: state['next_node'],
    {
        "what_find": "what_find",
        "call_llm": "call_llm",
        "off_topic": "off_topic",
        END: END,
    }
)


builder_for_graph.add_edge("what_find", "where_find")

builder_for_graph.add_edge("where_find", "retriever")

builder_for_graph.add_edge("retriever", "call_llm")

builder_for_graph.add_conditional_edges("call_llm", tools_condition)

builder_for_graph.add_edge("tools", "call_llm")


# Хранилище состояния отдельных диалогов
checkpointer = InMemorySaver()

# Долговоременная память о пользователе
store = InMemoryStore()

# компилируем граф
graph = builder_for_graph.compile(
    checkpointer=checkpointer,
    store=store,
)