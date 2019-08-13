# PersonalityDiscrimination_Chatting
Personality Representation &amp; Personality based Chatting   

This is a project of the "三个咕咕呆写出了一群bug"(Three heros & A bunch of bug) in Google AI ML Winter Camp. Thanks to my team members!  

[MBTI](https://en.wikipedia.org/wiki/Myers%E2%80%93Briggs_Type_Indicator) Personality can be judged by one's declaration or dialogue.   If a chatting machine can learn the speaker's personality, they might be able to make better dialogues.    
According to this idea, we make a simple chatting machine. Because we used a simple model and a few dialogue data, the result is not very good, but the idea of chatting machine with personality analysis may be worthwhile.     
There are another [repo](https://github.com/xiaotinghe/PCM) and [repo](https://github.com/shuhanfan/Personalize_Chatting_Bot) from my teamates.

<img src="https://github.com/Walleclipse/PersonalityDiscrimination_Chatting/raw/master/demo/chatbot1.png" width="400" >

More details shown in 'PersonalityDiscrimination_Chatting.pdf' 

## Method 

This work mainly consists of two parts, 1. Personality Discriminator: discriminate the speaker's personality according to some input sentences. 2. Chatting Machine: Generate the corresponding response according to the input sentence and the speaker's personality. 

<img src="https://github.com/Walleclipse/PersonalityDiscrimination_Chatting/raw/master/demo/model.png" width="800" >

1. Personality Discriminator:

The target of this part is classifying people into 16 distinct personality types across 4 axes, after feeding some dialogues or some declaration (or Twitter, Wechat …)  ([kaggle MBTI dataset](https://www.kaggle.com/datasnaek/mbti-type))

We used ELMo pre-trained model, Bi-LSTM encoder and self Attention Mechanisms.

2. Chatting Machine:

Given a post X= (x_1,x_2,..,x_n ) and a personality type p of the response
to be generated, the goal is to generate a response Y= (y_1,y_2,..,y_n )
that is coherent with the personality type p.

### About Code
`MBTI_discriminator_torch.py` ,  `MBTI_discriminator_bert.py` , `MBTI_discriminator_lgb.ipynb`         
All three files are MBTI discrimination model. The first file contains the main model.  
`dialogue.py`     
seq2seq conversation model   
`front`    
This folder contains the front-end program for chatting machine.    

## Results

The model outputs both personality scores and dialogue response.

<img src="https://github.com/Walleclipse/PersonalityDiscrimination_Chatting/raw/master/demo/chatbot2.jpg" width="300" >
