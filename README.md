# Rephrased Questions for the MP-DocVQA Dataset's Train Split

This repository contains 9 different rephrased questions for the [MP-DocVQA](https://rrc.cvc.uab.es/?ch=17&com=downloads) dataset's train split questions.

The data is stored in a pickle file and it contains a single dictionary. The keys are the question ids and the values are dictionaries with 2 key-value pairs:

 - 'original_question': A string containing the original question
 - 'augmented_questions': A list of 9 strings, containing the rephrased questions


These rephrased questions were created for my Master's thesis work, named as "**Can We Read a Book Without Opening It? A new Perspective Towards Multi-Page Visual Question Answering**". The Github repository with all the code will be available soon.



## Examples:

```

Question: What is the Amount Requested from PHS for Biology?  rephrased to: 
1.:  How much funding is sought from PHS for Biology?
2.:  What funding amount is being requested from PHS for Biology?
3.:  What is the sum requested from PHS for Biology?
4.:  What is the total amount requested from PHS for Biology purposes?
5.:  How much money is being asked from PHS for Biology?
6.:  What is the monetary amount requested from PHS for Biology?
7.:  What is the requested allocation from PHS for Biology?
8.:  What is the financial request to PHS for Biology?
9.:  What is the specified amount requested from PHS for Biology?

------------------------------------------------------------------------------

Question: What is the Offer Total for Vantage Travel Clock?  rephrased to: 
1.:  What is the total offer for the Vantage Travel Clock?
2.:  What is the Vantage Travel Clock's total offer?
3.:  How much is the total offer for Vantage Travel Clock?
4.:  What is the amount of the offer for Vantage Travel Clock?
5.:  What is the Vantage Travel Clock offer's total?
6.:  What is the total price of the offer for Vantage Travel Clock?
7.:  What is the Vantage Travel Clock's offer amount?
8.:  What is the total value of the offer for Vantage Travel Clock?
9.:  How much does the offer for Vantage Travel Clock total?

```






## Data Collection

The rephrased questions where obtained using the *llama3-70b-8192* model, using [Groq](https://groq.com/)'s API.

Due to the large number of questions which had to be rephrased, a different conversation was created for each. The used prompt contains instructions so that the rephrased questions contain all the key information stored in the original question, as well as some specific instructions to make the automatization process easier. This is the prompt which was used for obtaining the rephrased questions:

``` I am going to send you a question and I want you to rephrase it 9 times. The questions must be the same, but asked in different ways. And you can't refer to the person or people which the question is asked, so you can't rephrase it as 'Can you...' , 'Could you...'. You have to give the 9 rephrased questions separated by a single new line, so you have to return 9 lines, one rephrased question in each. Just write the rephrased questions, I do not want more verbose of any kind. Try to maintain as much keywords as possible from the original question. If there are names in the question they must appear in all rephrased questions. This is the question I want you to rephrase: ```



## How to use it:

The pickle file only occupies 15.58MB, so it can be loaded at once without any problem. Here is how it can be loaded:

``` 

import pickle as pkl
from typing import List

with open("./augmented_questions.pkl", "rb") as f:
    content = pkl.load(f)

EXAMPLE_QUESTION_ID = 29957

example_question_data = content[EXAMPLE_QUESTION_ID]
original_question: str = example_question_data['original_question']
rephrased_questions: List[str] = example_question_data['augmented_questions] 

```


## Contact

If you have any question about the rephrased questions you can contact me via:

- Mail: ialakunza@gmail.com
- LinkedIn: https://www.linkedin.com/in/i%C3%B1aki-lacunza-castilla-82b99527b/

