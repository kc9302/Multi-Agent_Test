from make_langgraph import activate_agent
import time


def main():
    korea_history = "이순신 장군의 절친은?"
    english = "영어로 코끼리는 뭐라고해요?"
    math = """위키쌤 이 문제를 초등학생 수학 계통도 안에서 설명해주세요.
가로의 길이가 세로의 길이보다 9 cm만큼 더 긴 직사각형의둘레의 길이가 50 cm일 때, 직사각형의 넓이를 구하시오.
x 표현은 사용하지 마세요.
되도록 미지수는 □, △로 말해주세요."""

    # 한국사 질문 예시
    start = time.time()
    text = activate_agent(question=korea_history)["answer"]
    end = time.time()
    print("\n------------ Answer ------------")
    for line in text.split("."):
        print(line)
    print(f"------------ 응답시간 : {end - start:.5f} sec ------------\n")

    # 영어 질문 예시
    start = time.time()
    result = activate_agent(question=english)["answer"]
    end = time.time()
    print("------------ Answer ------------")
    print(result)
    print()
    print(f"----------- 응답시간 : {end - start:.5f} sec ------------\n")

    # 수학 질문 예시
    start = time.time()
    text = activate_agent(question=math)["answer"]
    end = time.time()
    print("\n------------ Answer ------------")
    for line in text.split("."):
        print(line)
    print(f"----------- 응답시간 : {end - start:.5f} sec ------------")


if __name__ == "__main__":
    main()