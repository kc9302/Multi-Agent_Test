from langgraph.graph import StateGraph, START, END

from multi_agent import State, find_target, math, english, korea_history, retrieve, get_route


def graph_builder():    
    graph_build = StateGraph(State)

    # 노드 추가
    graph_build.add_node("Find_target", find_target)
    graph_build.add_node("Math", math)
    graph_build.add_node("English", english)
    graph_build.add_node("Korea_history", korea_history)
    graph_build.add_node("Retrieve", retrieve)
    graph_build.set_entry_point("Find_target")

    # 조건 엣지 설정
    graph_build.add_conditional_edges(
        "Find_target",
        get_route,
        {
        "수학":"Math",
        "영어":"English",
        "한국사":"Retrieve",
        }
    )

    graph_build.add_edge("Math", END)
    graph_build.add_edge("English", END)
    graph_build.add_edge("Retrieve", "Korea_history")
    graph_build.add_edge("Korea_history", END)

    # 그래프 생성
    graph = graph_build.compile()
    
    return graph

def activate_agent(question):
    return graph_builder().invoke({"messages": question})