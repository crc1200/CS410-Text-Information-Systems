<-------------------- Google Colab Link -------------------->

https://colab.research.google.com/drive/1HINqEIwIMyuxthpeLubtJxVbg1wlJuXq?usp=sharing

<-------------------- Link to Photos -------------------->

https://docs.google.com/document/d/1gy7azCG9ux31qYEad_S5MQxnJSypo_75MtGAhFPRJM0/edit?usp=sharing

<-------------------- 1 Word Prediction with Word2Vec and LSTM (50 points) -------------------->

    3.3 - Mean RMSE Values (Stopwords Removed)

        Mean RMSE (ReLU): 0.4380416
        Mean RMSE (Tanh): 0.43031064

        Histogram of RMSE When Removing Stopwords (ReLU vs Tanh) - [Image 1]

    3.5 - Words Closest to Predicted

        Top 5 similar words for predicted word vector 1:
        now (similarity: 0.8897)
        well (similarity: 0.8859)
        one (similarity: 0.8844)
        this (similarity: 0.8833)
        only (similarity: 0.8780)

        Top 5 similar words for predicted word vector 2:
        this (similarity: 0.8967)
        but (similarity: 0.8891)
        even (similarity: 0.8891)
        now (similarity: 0.8884)
        well (similarity: 0.8866)

    4 - Not Removing Stopwords

        Mean RMSE Values (Stopwords Not Removed)

        Mean RMSE (ReLU): 0.4160728
        Mean RMSE (Tanh): 0.41206867

        Predicted Words (Stopwords Not Removed - Tanh Model)

            Top 5 similar words for predicted word vector 1:
            this (similarity: 0.9196)
            one (similarity: 0.9137)
            but (similarity: 0.9117)
            it (similarity: 0.9116)
            well (similarity: 0.9111)

            Top 5 similar words for predicted word vector 2:
            it (similarity: 0.9172)
            this (similarity: 0.9165)
            so (similarity: 0.9153)
            even (similarity: 0.9105)
            well (similarity: 0.9097)

        Histogram of RMSE When Not Removing Stopwords (ReLU vs Tanh) - [Image 2]

    Conclusion

        When including stopwords, the RMSE distribution was more evenly distributed. In addition we saw a reduction in average RMSE values for both ReLU as well as Tanh. The reduction was around .02. Both points help show that including stopwords improves model performance, which is not what I would have expected. 

<-------------------- 2 Word Prediction with Transformers - (30 points) -------------------->

    Comparing BERT Accuracy with and without EOS

        BERT Accuracy (Removing Punctuation)

            0.487%

        BERT Accuracy (Adding EOS Punctuation)

            28.039%

    GPT2 Accuracy (EOS Not Tested because only using one sentence)

        GPT2_Accuracy

            23.566%

    Cosine Similarity Between BERT and GPT-2

        It is difficult to say which model worked better overall when strictly looking at the cosine similarity graph. The BERT graph is more evenly distributed, though has greater frequency for complete matches. However the GPT model has more concentration further to the right. Overall though, it would appear that BERT is the better performing model, though only when including EOS punctuation. From an accuracy perspective, BERT (when including EOS punctuation) was also the more accurate model, with roughly a 4.5% increase in accuracy. 

<-------------------- 3 Topic Modeling - Multinomial PMM Model and LDA(20 points) -------------------->

Part A

    Top Five Words (Topic 7)

        said: 0.0220
        us: 0.0205
        reuters: 0.0197
        new: 0.0183
        oil: 0.0139

Part B

    [Image 4]