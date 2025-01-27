import random
from typing import Union

instruct_template = {
    "prompt_input": "아래는 작업을 설명하는 지침과 추가 입력을 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 답변을 작성해주세요.\n\n### 지침:\n{instruction}\n\n### 입력:\n{input}\n\n### 답변:",
    "prompt_no_input" : "아래는 작업을 설명하는 지침입니다. 요청을 적절히 완료하는 답변을 작성해주세요.\n\n### 지침:\n{instruction}\n\n### 답변:\n",
    "response_split": "### 답변:"
}

few_shot_input_text = [
    '잔소리 직장 상사 회사가 사무실 오히려 좋아', '정형돈 현기증난단말이에요 좀만 버티다가 쓰러져야 되겠다 하기싫어 극한알바',
    '유희열 서른살 반사회적 나이 독신 솔로의 놀림 안습 30살', '유재석 하지마 그 입을 더 열지마 쉿 분노 노래를부르고있다.',
    '무한상사 사직 회사 직장 출근 사무실 상사 꿈나라 희망 안녕히계세요 안녕 계세요', '정준하 사는데 지장 없어요 불만제로',
    '너무 예쁜 아름다워 훌륭해요 칭찬'
]

few_shot_output_text = [{'감정' : '좋음', '상황' : '직장에서 상사의 잔소리를 받는 상황',
             '상황 유형' : '감정 표현', '의도' : '상사의 잔소리를 긍정적으로 받아들임.',
             '최종 상황' : '상사의 잔소리를 긍정적으로 받아들여 직장에서 긍정적인 변화를 이끌어냄'},
          {'감정' : '절망, 힘듦', '상황' : '누군가가 현기증과 지침에 시달리며 극한 알바 상황에 처한 상황',
           '상황 유형' : '자기 반영', '의도' : '자신의 고통스러운 상황을 표현하며 조금만 하고 쓰러지겠다는 마음을 나타냄.',
           '최종 상황' : '누군가가 현기증과 지침에 시달리며, 조금만 하고 쓰러지고자 하는 상황을 표현함.'},
          {'감정' : '슬픔', '상황' : '독신으로 사는 것을 놀리는 상황',
           '상황 유형' : '놀림', '의도' : '많은 나이에 독신으로 사는 것에 대해 놀리고자 함.',
           '최종 상황' : '누군가가 상대방이 독신으로 사는 것에 대하여 놀리는 상황'},
          {'감정' : '분노', '상황' : '누군가가 다른 사람에게 말하며 분노의 감정을 표현하고 있는 상황',
            '상황 유형' : '요구', '의도' : '상대방이 말하는 것을 말리고자 함',
            '최종 상황' : '상대방에게 분노를 표현하며 그의 행동을 멈추고자 함.'},
           {'감정' : '행복', "상황" : "누군가가 회사를 그만두고 나가는 상황",
           '상황 유형' : '자기 반영', "의도" : '회사를 그만둬서 행복함을 표현',
           '최종 상황' : '직장을 그만두어 매우 행복함을 표현함.'},
          {'감정' : '만족', '상황' : '누군가가 현재의 생활에 만족하는 상황',
           '상황 유형' : '자기 반영', '의도' : '자신의 현재 상태에 대한 만족을 표현하며 긍정적인 감정을 나타내고자 함',
           '최종 상황' : '누군가가 현재의 생활에 만족하며, 불만이 없는 상태를 나타내고 있음'},
          {'감정': '기쁨', '상황': '누군가가 아름답다는 칭찬을 받아 기쁨을 느끼는 상황',
          '상황 유형': '타인 반영', '의도' : '상대방을 칭찬하며, 그들의 아름다움을 인정하고자 함.',
           '최종 상황' : '누군가가 상대방을 칭찬하며 그들의 아름다움을 인정하고 있음.'
           }]

few_shot_query_input_text = ['너 조용히 해봐', '되는 일이 하나도 없네', '젠장... 회사에서 또 끌고 왔어', '드디어 회사 그만뒀다~~']

few_shot_query_output_text = [{'감정' : '분노', '상황' : '상대를 조용히 시키는 상황',
                 '문장의 유형' : '요구', '의도' : '상대방을 조용하게 하고자 함.',
                 '최종 상황' : '상대방에게 분노를 표현하여 그의 행동을 조용하게 하고자 함.'},
                {'감정' : '절망', '상황' : '누군가가 실패와 무기력함을 느끼며 일상에 좌절한 상황',
                 '문장의 유형' : '자기 반영', '의도' : '자신이 겪고 있는 어려운 상황에 좌절하였음을 표현',
                 '최종 상황' : '누군가가 자신의 절망과 무기력함 표현'},
                {'감정' : '짜증', '상황' : '누군가가 회사에 끌려간 상황',
                 '문장의 유형' : '감정 표현', '의도' : '불쾌한 끌려옴에 대한 짜증을 표현하고자 함.',
                 '최종 상황' : '누군가가 회사로부터의 불쾌한 끌려옴에 짜증난 상황'},
                {'감정' : '해방', '상황' : '누군가가 회사를 그만둔 상황',
                 '문장의 유형' : '성취감 표현', '의도' : '자신의 성취를 표현하여 기쁨을 나타내고자 함.',
                 '최종 상황' : '누군가가 회사를 그만두며, 성취를 표현하며 해방된 기분을 느끼고 있음.'}]

def get_query_prompt():
    prompt_dict = {}

    prompt_dict['instruction'] = '문장이 하나가 주어집니다. 문장을 바탕으로 발화자의 감정, 상황, 문장 유형, 의도를 예측하고, 그것들을 토대로 최종상황을 예측해줘.'
    for i in range(len(few_shot_query_input_text)):
        prompt_dict['instruction'] += f'\n\n### 입력 :\n{few_shot_query_input_text[i]}\n\n### 답변:'
    for key, value in few_shot_query_output_text[i].items():
        prompt_dict['instruction'] += f'\n{key} : {value}'

    return prompt_dict

def get_document_prompt():
    prompt_dict = {}
    prompt_dict['instruction'] = '공백으로 단어들이 구분되어 있습니다. 단어들을 바탕으로 감정, 상황, 상황 유형, 의도를 예측하고, 그것들을 토대로 최종상황을 예측해줘. 단, 고유명사는 예측에서 빼보자'

    input_random = random.sample(range(7), 4)

    for j in input_random:
        prompt_dict['instruction'] += f'\n\n### 입력 :\n{few_shot_input_text[j]}\n\n### 답변:'
        for key, value in few_shot_output_text[j].items():
            prompt_dict['instruction'] += f'\n{key} : {value}'

def get_rerank_prompt(query_input, meme_prompt, top):

    instruct_template = {
        "prompt_input" :  "아래는 작업을 설명하는 지침과 추가 입력을 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 답변을 작성해주세요.\n\n### 지침:\n{instruction}\n\n### 입력:\n{input}\n\n### 답변:\n",
    }


    prompt_dict = {}
    prompt_dict['instruction'] = '한국어로 어떤 상황이 주어지면 아래에서 미리 알려줄 밈 태그들 중에 적절한 것을 골라봅시다. \n\n밈 태그들은 다음과 같이 존재합니다.\n\n \n\nID0 : 입닫자 입닫아라 주둥이 닥쳐 입입입 입 정준하 노홍철 무도 무한도전\nID1 : 감사 무도 무한도전 절하는 인사 단체 큰절 감사합니다\nID2 : 아니 아닌데 정형돈 무한도전 무도\nID3 : 무도짤 박명수 상식적으로 이해할 수 없는 행동 이해불가 황당 상식\nID4 : 밥 갖고 와 노홍철 밥줘 상남자 무한도전 화내는 무도\nID5 : 홍진경 아니요 지쳤나요 지금 무표정 무도 무한도전 지금지쳤나요\nID6 : 무도짤 정형돈 메모 지친 피곤한\nID7 : 시무룩 좌절 박명수 무도 무한도전 표정 슬픈\nID8 : 하기 싫어 박명수 무도짤 무한상사 싫다 짜증\nID9 : 정형돈 무한도전 무도 얌전한 사람을 거칠게 만들어요 열받아 화나게 화남 화나\n\n\n### 입력 :\n대화 상황이 "아유 아무것도 하기 싫다."일 때 이와 가장 비슷한 상황의 밈 태그의 IDX는 무엇일까요? \n\n### 답변 : ID8 \n\n밈 태그들은 다음과 같이 존재합니다.\n\n'

    for i, _ in top:
        prompt_dict['instruction'] += f'IDX {i} : ' + meme_prompt[i] + '\n'

    prompt_dict['input'] = f'대화 상황이 "{query_input}"일 때 이와 가장 비슷한 상황의 밈 태그의 IDX는 무엇일까요?'

    prompt_input = instruct_template["prompt_input"].format(instruction = prompt_dict['instruction'], input = prompt_dict['input'])

    return prompt_input



class Prompter(object):

    def __init__(self, verbose: bool = False):
        self.template = instruct_template

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:

        if input: # input text가 있다면
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )

        if label:
            res = f"{res}{label}"

        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()