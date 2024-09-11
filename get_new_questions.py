

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from dataset import getDataloaders

from groq import Groq

from tqdm import tqdm
from typing import List, Dict
import re
import pickle as pkl
import random

from api_key import KEY




INITIAL_MESSAGE = "I am going to send you a question and I want you to rephrase it 9 times.\
        The questions must be the same, but asked in different ways. And you can't refer to the \
        person or people which the question is asked, so you can't rephrase it as 'Can you...' , 'Could you...'.\
        You have to give the 9 rephrased questions separated by new lines. \
        Just write the rephrased questions, I do not want more verbose of any kind. \
        Try to maintain as much keywords as possible from the original question. \
        If there are names in the question they must appear in all rephrased questions. \
        This is the question I want you to rephrase: "
    
        # Important keywords must be wirtten
        # Names must be kept, otherwise N Anand was not used, for instance
        # If the 'Can you... , Could you...' prohibition was not specified then it
        # asked the questions as if we were referring to an assistant



def test():

    client = Groq(api_key=KEY)
    question = "How many number of options are granted to N Anand during the financial year?"
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "user",
                "content": INITIAL_MESSAGE + question
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")




class questionRephraser():
    '''
    Rephrase given questions using Llama3-70b-8192, using GROQ's API
    In order to use GROQ, an account must be created and a API-KEY
    obtained. Once the API-KEY has been obtained, in the terminal 
    the following line should be written
    export GROQ_API_KEY=<your-api-key-here>
    '''

    def __init__(self,
                 n_rephrased_questions: int = 9,
                 initial_message: str = None,
                 save_path: str = "./augmented_questions.pkl",
                 save_pkl_every_n_generations: int = 10):
        super().__init__()

        train_dataloader, _ = getDataloaders(train_batch_size=1,
                                             val_batch_size=1,
                                             num_workers=2)
        self.train_dataloader = train_dataloader

        self.save_path = save_path
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                self.augmented_questions: Dict = pkl.load(f)
        else:
            raise ValueError("Not loaded previous pickle!")
            self.augmented_questions: Dict = {}

        self.save_pkl_every_n_generations = save_pkl_every_n_generations

        if not initial_message:
            self.n_rephrased_questions = n_rephrased_questions

            self.initial_message = f"I am going to send you a question and I want you to rephrase it {self.n_rephrased_questions} times.\
                The questions must be the same, but asked in different ways. And you can't refer to the \
                person or people which the question is asked, so you can't rephrase it as 'Can you...' , 'Could you...'.\
                You have to give the 9 rephrased questions separated by a single new line, \
                so you have to return 9 lines, one rephrased question in each. \
                Just write the rephrased questions, I do not want more verbose of any kind. \
                Try to maintain as much keywords as possible from the original question. \
                If there are names in the question they must appear in all rephrased questions. \
                This is the question I want you to rephrase: "
            
            self.insist_message = f"No, you have to give the 9 rephrased questions separated by new lines. \
                You must insert a prefix which I can take off with the following pattern: \
                prefix_pattern = re.compile(r'^\s*Here are the \d+ rephrased questions:\s*', re.IGNORECASE) \
                Then I get then aswer using this python line: 'answer = re.sub(prefix_pattern, '', answer)' \
                then I use this python line to separate the rephrased questions: 'questions = re.split(r'\n+', answer.strip())' \
                and finnally I clean them using 'cleaned_questions = [q.strip() for q in questions if q.strip()]' . \
                SO YOU HAVE TO GIVE ME THE REPHRASED QUESTIONS EXACTLY AS I HAVE TOLD YOU. \
                Here is again the question I want you to rephrase 9 times: "

    def _ask_rephrasing(self, 
                        question: str) -> str:
        client = Groq(api_key=KEY)
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "user",
                    "content": self.initial_message + question
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        answer = ""
        for chunk in completion:
            answer += chunk.choices[0].delta.content or ""

        return answer


    def _insist_rephrasing(self, 
                           question: str) -> List[str]:
        
        client = Groq(api_key=KEY)
        message_list = [
                {
                    "role": "user",
                    "content": self.initial_message + question
                }
            ]
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=message_list,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        answer = ""
        for chunk in completion:
            answer += chunk.choices[0].delta.content or ""
        rephrased_questions: List[str] = self._organize_answer(answer)
        if len(rephrased_questions) != 9:
            while len(rephrased_questions) != 9:
                print(f"Insisting..., answer: {answer}\n\n")
                message_list.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )
                message_list.append(
                    {
                        "role": "user",
                        "content": self.insist_message + question
                    }
                )
                completion = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=message_list,
                    temperature=1.5,
                    max_tokens=1024,
                    top_p=1,
                    stream=True,
                    stop=None,
                )
                answer = ""
                for chunk in completion:
                    answer += chunk.choices[0].delta.content or ""
                rephrased_questions = self._organize_answer(answer)

        return rephrased_questions



    def _organize_answer(self,
                         answer: str
                         ) -> List[str]:
        # Use a regular expression to find and remove the prefix
        prefix_pattern = re.compile(r"^\s*Here are the \d+ rephrased questions:\s*", re.IGNORECASE)
        answer = re.sub(prefix_pattern, '', answer)
        
        # Split the remaining text into individual questions, handling multiple newlines
        questions = re.split(r'\n+', answer.strip())
        
        # Clean up each question and remove any extra whitespace
        cleaned_questions = [q.strip() for q in questions if q.strip()]
        
        return cleaned_questions


    def get_rephrased_questions(self,
                                question: str
                                ) -> List[str]:
        
        answer: str = self._ask_rephrasing(question)
        rephrased_questions: List[str] = self._organize_answer(answer)
        
        if len(rephrased_questions) != 9:
            try:
                rephrased_questions: List[str] = self._insist_rephrasing(question)
            except:
                print(f"question: {question} could not be rephrased")
        
        #assert len(rephrased_questions) == 9 , "The organizing is not correct"

        return rephrased_questions
    

    def rephrase_training_set_questions(self):
        progress_bar = tqdm(self.train_dataloader, desc="Training epoch")
        counter = 0
        for sample_idx, sample in enumerate(progress_bar):
            
            question_id = sample['question_id'][0]
            if self.augmented_questions[question_id] is not None:
                print(f"Augmented questions already generated for: {question_id}  , skipping....")
                continue

            original_question: str = sample['question'][0]
            rephrased_questions: List[str] = self.get_rephrased_questions(original_question)
            if not rephrased_questions: continue

            all_questions: List[str] = [original_question] + rephrased_questions

            self.augmented_questions[question_id] = {
                "original_questions": original_question,
                "augmented_questions": all_questions
            }

            print("\n============================")
            print(f"Question: {original_question}  rephrased to: ")
            for i, aug_question in enumerate(rephrased_questions):
                print(f"{i+1}.:  {aug_question}")
            print("=============================\n")

            counter += 1
            if counter == self.save_pkl_every_n_generations:
                with open(self.save_path, "wb") as save_f:
                    pkl.dump(self.augmented_questions, save_f)
                counter = 0
                print("pkl file saved")

        with open(self.save_path, "wb") as save_f:
            pkl.dump(self.augmented_questions, save_f)  



def main() -> None:

    rephraser = questionRephraser(n_rephrased_questions=9,
                                  initial_message=None,
                                  save_path="./augmented_questions.pkl")

    # TEST
    # original_question = "How many number of options are granted to N Anand during the financial year?"
    # rephrased_question = rephraser.get_rephrased_questions(question=original_question)
    # print(rephrased_question)

    rephraser.rephrase_training_set_questions()


if __name__ == '__main__':
    main()










